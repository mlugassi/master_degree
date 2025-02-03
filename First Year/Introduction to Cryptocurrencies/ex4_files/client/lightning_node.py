from typing import Dict, Optional, List, Any
from client.utils import APPEAL_PERIOD, ChannelStateMessage, EthereumAddress, IPAddress, PrivateKey, Signature
from hexbytes import HexBytes
from eth_typing import HexAddress, HexStr


from client.network import Message, Network
from client.node import Node
from web3 import Web3
from .utils 


class LightningNode(Node):
    """represents a payment channel node that can support several payment channels."""

    def __init__(self, private_key: PrivateKey, eth_address: EthereumAddress, networking_interface: Network, ip: IPAddress, w3: Web3, contract_bytecode: str, contract_abi: Dict[str, Any]) -> None:
        """Creates a new node that uses the given ethereum account (private key and address),
        communicates on the given network and has the provided ip address. 
        It communicates with the blockchain via the supplied Web3 object.
        It is also supplied with the bytecode and ABI of the Channel contract that it will deploy.
        All values are assumed to be legal."""
        self._private_key = private_key
        self._w3 = w3
        self._contract_bytecode = contract_bytecode
        self._contract_abi = contract_abi
        self._network = networking_interface
        self._ip_address = ip
        self._eth_address = eth_address
        # Maintain an internal list/dict of channels.
        self._channels: Dict[EthereumAddress, ChannelStateMessage] = {}
        self._channels_other_party: Dict[EthereumAddress, IPAddress] = {}

    def get_list_of_channels(self) -> List[EthereumAddress]:
        """returns a list of channels managed by this node. The list will include all open channels,
        as well as closed channels that still have the node's money in them.
        Channels are removed from the list once funds have been withdrawn from them."""
        return list(self._channels.keys())

    def establish_channel(self, other_party_eth_address: EthereumAddress, other_party_ip_address: IPAddress,  amount_in_wei: int) -> EthereumAddress:
        """Creates a new channel that connects the address of this node and the address of a peer.
        The channel is funded by the current node, using the given amount of money from the node's address.
        returns the address of the channel contract. Raises a ValueError exception if the amount given is not positive or if it exceeds the funds controlled by the account.
        The IPAddress and ethereum address of the other party are assumed to be correct."""
        if amount_in_wei <= 0 or amount_in_wei > self._w3.eth.get_balance(self.eth_address):
            raise ValueError("Amount must be positive.")
        
        contract = self._w3.eth.contract(abi=self._contract_abi, bytecode=self._contract_bytecode)
        tx_hash = contract.constructor(other_party_eth_address, APPEAL_PERIOD).transact({ 'from': self.eth_address, 'value': amount_in_wei })
        tx_receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)
        deployed_channel_address = EthereumAddress(self._w3.toChecksumAddress(tx_receipt.contractAddress))

        # Create an initial state: this node's balance is the funded amount, the other node's balance is 0.
        initial_state = ChannelStateMessage(
            channel_address=deployed_channel_address,
            balance1=amount_in_wei,
            balance2=0,
            serial_number=0,
            # sig=Signature((0, sign(self.private_key, ), "")) #TODO need to understnad all the keta with the signaure here
        )
        self._channels[deployed_channel_address] = initial_state
        self._channels_other_party[deployed_channel_address] = other_party_ip_address

        self._network.send_message(other_party_ip_address, Message.NOTIFY_OF_CHANNEL, deployed_channel_address, self.ip_address)

        return deployed_channel_address

    @property
    def eth_address(self) -> EthereumAddress:
        """returns the ethereum address of this node"""
        return self._eth_address

    @property
    def ip_address(self) -> IPAddress:
        return self._ip_address

    @property
    def private_key(self) -> PrivateKey:
        """returns the private key of this node"""
        return self._private_key

    def send(self, channel_address: EthereumAddress, amount_in_wei: int) -> None:
        """sends money in one of the open channels this node is participating in and notifies the other node.
        This operation should not involve the blockchain.
        The channel that should be used is identified by its contract's address.
        If the balance in the channel is insufficient, or if a node tries to send a 0 or negative amount, raise an exception (without messaging the other node).
        If the channel is already closed, raise an exception."""
        if amount_in_wei <= 0:
            raise ValueError("Amount must be positive.")
        if channel_address not in self._channels:
            raise ValueError("Channel does not exist.")
        # For simplicity, assume channel state is updated by subtracting from our balance.
        state = self._channels[channel_address]
        if state.balance1 < amount_in_wei:
            raise ValueError("Insufficient balance in channel.")
        # Update dummy state: reduce our balance, increase counter, etc.
        new_state: ChannelStateMessage = ChannelStateMessage()
        new_state.contract_address = channel_address
        new_state.balance1 = state.balance1 - amount_in_wei
        new_state.balance2 = state.balance2 - amount_in_wei
        new_state.serial_number = state.serial_number + 1

        new_state.sig = Signature(0, sing(self.private_key, new_state.message_hash), '')
        self._channels[channel_address] = new_state
        self._network.send_message(self._channels_other_party[channel_address], Message.NOTIFY_OF_CHANNEL, new_state, self.ip_address)


    def get_current_channel_state(self, channel_address: EthereumAddress) -> ChannelStateMessage:
        """
        Gets the latest state of the channel that was accepted by the other node
        (i.e., the last signed channel state message received from the other party).
        If the node is not aware of this channel, raise an exception.
        """
        if channel_address not in self._channels:
            raise Exception("Channel not found.")
        return self._channels[channel_address]
    
    def close_channel(self, channel_address: EthereumAddress, channel_state: Optional[ChannelStateMessage] = None) -> bool:
        """
        Closes the channel at the given contract address.
        If a channel state is not provided, the node attempts to close the channel with the latest state that it has,
        otherwise, it uses the channel state that is provided (this will allow a node to try to cheat its peer).
        Closing the channel begins the appeal period automatically.
        If the channel is already closed, throw an exception.
        The other node is *not* notified of the closed channel.
        If the transaction succeeds, this method returns True, otherwise False."""
        if channel_address not in self._channels:
            raise Exception("Channel not found or already closed.")
        # For simulation, we simply remove the channel from our active list.
        # If a channel_state is provided, we could simulate a cheating attempt.
        del self._channels[channel_address]
        return True
    
    def appeal_closed_chan(self, contract_address: EthereumAddress) -> bool:
        """
        Checks if the channel at the given address needs to be appealed, i.e., if it was closed with an old channel state.
        If so, an appeal is sent to the blockchain.
        If an appeal was sent, this method returns True. 
        If no appeal was sent (for any reason), this method returns False.
        """
        # Dummy implementation: always return True if the channel was closed with an old state.
        # In real life, would check timestamps and state serial numbers.
        return True
    
    def withdraw_funds(self, contract_address: EthereumAddress) -> int:
        """allows the user to claim the funds from the channel.
        The channel needs to exist, and be after the appeal period time. Otherwise an exception should be raised.
        After the funds are withdrawn successfully, the node forgets this channel (it no longer appears in its open channel lists).
        If the balance of this node in the channel is 0, there is no need to create a withdraw transaction on the blockchain.
        This method returns the amount of money that was withdrawn (in wei)."""
        # In a real scenario, funds would be transferred from the channel contract.
        # Here, we simulate by returning a dummy value and removing the channel.
        if contract_address not in self._channels:
            raise Exception("Channel not found or already withdrawn.")
        state = self._channels[contract_address]
        # Assume our balance is state.balance1; if zero, nothing to withdraw.
        amount = state.balance1
        del self._channels[contract_address]
        return amount
    
    def notify_of_channel(self, contract_address: EthereumAddress, other_party_ip_address: IPAddress) -> None:
        """This method is called to notify the node that another node created a channel in which it is participating.
        The contract address for the channel is provided.

        The message is ignored if:
        1) This node is already aware of the channel
        2) The channel address that is provided does not involve this node as the second owner of the channel
        3) The channel is already closed
        4) The appeal period on the channel is too low
        For this exercise, there is no need to check that the contract at the given address is indeed a channel contract (this is a bit hard to do well)."""
        # Simulate notification: if not already known, add a dummy channel state.
        if contract_address in self._channels:
            return  # already known
        dummy_state = ChannelStateMessage(
            # contract_address, 0, 0, 0, Signature((0, \"\", \"\"))
        )
        self._channels[contract_address] = dummy_state

    def ack_transfer(self, msg: ChannelStateMessage) -> None:
        """This method receives a confirmation from another node about the transfer.
        The confirmation is supposed to be a signed message containing the last state sent to the other party,
        but now signed by the other party. In fact, any message that is signed properly, with a larger serial number,
        and that does not strictly decrease the balance of this node, should be accepted here.
        If the channel in this message does not exist, or the message is not valid, it is simply ignored."""

    def receive_funds(self, state_msg: ChannelStateMessage) -> None:
        """A method that is called when this node receives funds through the channel.
        A signed message with the new channel state is receieved and should be checked. If this message is not valid
        (bad serial number, signature, or amounts of money are not consistent with a transfer to this node) then this message is ignored.
        Otherwise, the same channel state message should be sent back, this time signed by the node as an ACK_TRANSFER message.
        """
