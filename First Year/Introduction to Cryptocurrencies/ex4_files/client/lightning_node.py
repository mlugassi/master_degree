from typing import Dict, Optional, List, Any
from client.utils import APPEAL_PERIOD, ChannelStateMessage, EthereumAddress, IPAddress, PrivateKey, Signature
from hexbytes import HexBytes
from eth_typing import HexAddress, HexStr


from client.network import Message, Network
from client.node import Node
from web3 import Web3
from .utils import *


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
        # self._eth_account = HasEthAccount(eth_address=eth_address, private_key=private_key)
        # Maintain an internal list/dict of channels.
        self._channels: Dict[EthereumAddress, Channel] = {}

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
        
        contract: Contract = Contract.deploy(self._w3, self._contract_bytecode, self._contract_abi, self, 
                                   ctor_args=(other_party_eth_address, APPEAL_PERIOD), deploy_kwargs={ 'from': self.eth_address, 'value': amount_in_wei })
        
        # Create an initial state: this node's balance is the funded amount, the other node's balance is 0.
        initial_state = sign(self._private_key, ChannelStateMessage(
            contract_address=contract.address,
            balance1=amount_in_wei,
            balance2=0,
            serial_number=0
        ))
        
        self._channels[contract.address] = Channel(pending_states=[],
                                                   cur_state=initial_state,
                                                   other_eth_address=other_party_eth_address,
                                                   other_ip=other_party_ip_address,
                                                   contract=contract)
             
        self._network.send_message(self._channels[contract.address].other_ip, Message.NOTIFY_OF_CHANNEL, contract.address, self._ip_address)

        return contract.address

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

    def get_my_balance(self, channel_address: EthereumAddress) -> int:
        cur_state = self.get_current_channel_state(channel_address)
        if cur_state:
            return cur_state.balance1 if self.am_i_first_owner(cur_state.contract_address) else cur_state.balance2
        return 0    
        
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
        if self.get_my_balance(channel_address) < amount_in_wei:
            raise ValueError("Insufficient balance in channel.")

        state = self.get_current_channel_state(channel_address)

        i_first_owner = self.am_i_first_owner(channel_address)
        new_state: ChannelStateMessage = sign(self._private_key, ChannelStateMessage(
            contract_address=channel_address,
            balance1=state.balance1 - amount_in_wei if i_first_owner else state.balance1 + amount_in_wei,
            balance2=state.balance2 + amount_in_wei if i_first_owner else state.balance2 - amount_in_wei,
            serial_number=state.serial_number + 1,
        ))

        self._channels[channel_address].pending_states.append(new_state)
        self._network.send_message(self._channels[channel_address].other_ip, Message.RECEIVE_FUNDS, new_state)

    def get_current_channel_state(self, channel_address: EthereumAddress) -> ChannelStateMessage:
        """
        Gets the latest state of the channel that was accepted by the other node
        (i.e., the last signed channel state message received from the other party).
        If the node is not aware of this channel, raise an exception.
        """
        if channel_address not in self._channels:
            raise Exception("Channel not found.")
        return self._channels[channel_address].cur_state # What happend if cur_state is None
    
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
            #raise Exception("Channel not found.") # TODO check if we should raise an exception or return False
            return False
        
        if get_channel_state(self._channels[channel_address].contract) != State.OPEN:
            raise Exception("You can only close an open channel.")
        
        if channel_state is None:
            channel_state = self.get_current_channel_state(channel_address)
            
        return close_one_side(self._channels[channel_address].contract, self, channel_state) #TODO there is a case that we will call that func with cur_state = None
    
    def appeal_closed_chan(self, contract_address: EthereumAddress) -> bool:
        """
        Checks if the channel at the given address needs to be appealed, i.e., if it was closed with an old channel state.
        If so, an appeal is sent to the blockchain.
        If an appeal was sent, this method returns True. 
        If no appeal was sent (for any reason), this method returns False.
        """
        if contract_address not in self._channels:
            return False
        
        if get_channel_state(self._channels[contract_address].contract) != State.APPEAL_PERIOD:
            return False

        cur_state = self.get_current_channel_state(contract_address)
        if cur_state and get_serial_num(self._channels[contract_address].contract) >= cur_state.serial_number:
            return False
        
        return appeal_closure(self._channels[contract_address].contract, self, cur_state) #TODO there is a case that we will call that func with cur_state = None
    
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
        
        if get_channel_state(self._channels[contract_address].contract) != State.CLOSE:
            raise Exception("Channel not closed.")
        
        my_balance =  get_balance(self._channels[contract_address].contract, self.eth_address)
        if my_balance > 0:
            withdraw(self._channels[contract_address].contract, self, self.eth_address)
    
        del self._channels[contract_address]
        return my_balance
    
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
        contract = Contract(contract_address, self._contract_abi, self._w3)
        
        if contract_address in self._channels: # 1.
            return None
        
        if get_first_owner(contract) == self.eth_address or get_other_owner(contract) != self.eth_address: # 2.
            return None
        
        if get_channel_state(contract) == State.CLOSE: # 3.
            return None

        if get_appeal_period_len(contract) < APPEAL_PERIOD: # 4.
            return None

        # initial_state = sign(self._private_key, ChannelStateMessage(
        #     channel_address=contract_address,
        #     balance1=contract.get_balance1(),
        #     balance2=0,
        #     serial_number=0
        # ))

        self._channels[contract_address] = Channel(pending_states=[],
                                                #    cur_state=initial_state,
                                                   cur_state=None,
                                                   other_eth_address=get_first_owner(contract),
                                                   other_ip=other_party_ip_address,
                                                   contract=contract)

    def get_pending_state(self, msg: ChannelStateMessage) -> Optional[ChannelStateMessage]:
        for state in self._channels[msg.contract_address].pending_states:
            if state.message_hash == msg.message_hash:
                return state       
        return None
                                        
    def ack_transfer(self, msg: ChannelStateMessage) -> None:
        """This method receives a confirmation from another node about the transfer.
        The confirmation is supposed to be a signed message containing the last state sent to the other party,
        but now signed by the other party. In fact, any message that is signed properly, with a larger serial number,
        and that does not strictly decrease the balance of this node, should be accepted here.
        If the channel in this message does not exist, or the message is not valid, it is simply ignored."""
        if msg.contract_address not in self._channels:
            return None
        
        if not validate_signature(msg, self._channels[msg.contract_address].other_eth_address):
            return None
        
        if msg.balance1 < 0 or msg.balance2 < 0:
            return None
        
        cur_state = self.get_current_channel_state(msg.contract_address)
        if cur_state:
            if cur_state.serial_number >= msg.serial_number:
                return None
                
            if (msg.balance1 + msg.balance2) != (cur_state.balance1 + cur_state.balance2):
                return None
        
        if self.get_pending_state(msg) is None:
            return None
        
        self._channels[msg.contract_address].cur_state = msg
        self.clean_unrelevant_states(msg.contract_address)

    def receive_funds(self, state_msg: ChannelStateMessage) -> None:
        """A method that is called when this node receives funds through the channel.
        A signed message with the new channel state is receieved and should be checked. If this message is not valid
        (bad serial number, signature, or amounts of money are not consistent with a transfer to this node) then this message is ignored.
        Otherwise, the same channel state message should be sent back, this time signed by the node as an ACK_TRANSFER message.
        """
        if state_msg.contract_address not in self._channels:
            return None
        
        if not validate_signature(state_msg, self._channels[state_msg.contract_address].other_eth_address):
            return None
        
        if state_msg.balance1 < 0 or state_msg.balance2 < 0:
            return None
                        
        cur_state = self.get_current_channel_state(state_msg.contract_address)
        if cur_state:
            if state_msg.serial_number <= cur_state.serial_number:
                return None

            if (state_msg.balance1 + state_msg.balance2) != (cur_state.balance1 + cur_state.balance2):
                return None
            
            if (self.am_i_first_owner(cur_state.contract_address) and state_msg.balance1 < cur_state.balance1) or \
                (not self.am_i_first_owner(cur_state.contract_address) and state_msg.balance2 < cur_state.balance2):
                return None
        else:
            cur_state = state_msg
        
        self._channels[cur_state.contract_address].cur_state = state_msg
        self.clean_unrelevant_states(cur_state.contract_address)

        self._network.send_message(self._channels[state_msg.contract_address].other_ip, Message.ACK_TRANSFER, sign(self.private_key, ChannelStateMessage(
            contract_address=state_msg.contract_address,
            serial_number=state_msg.serial_number,
            balance1=state_msg.balance1,
            balance2=state_msg.balance2
        ),))

    def am_i_first_owner(self, channel_address: EthereumAddress) -> bool:
        return get_first_owner(self._channels[channel_address].contract) == self.eth_address

    def clean_unrelevant_states(self, channel_address: EthereumAddress) -> None:
        for state in self._channels[channel_address].pending_states.copy():
            if state.serial_number <= self.get_current_channel_state(channel_address).serial_number:
                self._channels[channel_address].pending_states.remove(state)

class Channel:
    def __init__(self, cur_state: ChannelStateMessage, pending_states: list[ChannelStateMessage], other_eth_address: EthereumAddress, other_ip:IPAddress, contract:Contract ) -> None:
        self.cur_state: ChannelStateMessage = cur_state
        self.pending_states: list[ChannelStateMessage] = pending_states
        self.other_eth_address: EthereumAddress = other_eth_address
        self.other_ip: IPAddress = other_ip
        self.contract: Contract = contract

class State:
    OPEN = 0
    APPEAL_PERIOD = 1
    CLOSE = 2



def get_first_owner(contract) -> EthereumAddress:
    """Returns the first owner of the contract."""
    return contract.call("getFirstOwner")

def get_other_owner(contract) -> EthereumAddress:
    """Returns the second owner of the contract."""
    return contract.call("getOtherOwner")

def get_channel_state(contract) -> State:
    channel_state = contract.call("getChannelState")
    if channel_state == 0:
        return State.OPEN
    if channel_state == 1:
        return State.APPEAL_PERIOD
    if channel_state == 2:
        return State.CLOSE           

def get_balance(contract, my_address: EthereumAddress) -> int:
    return contract.call("getBalance", call_kwargs={ 'from': my_address })

def get_appeal_period_len(contract) -> int:
    return contract.call("getAppealPeriodLen")

def get_serial_num(contract) -> int:
    return contract.call("getSerialNum")
    
def withdraw(contract, user: HasEthAccount, dest_address: EthereumAddress) -> None:
    # contract.transact(user, "withdrawFunds", func_args=tuple(Web3.to_checksum_address(dest_address))) #TODO check why Ofic func is not working for it (it spliting the address to 42 strings)
    tx = contract._contract.functions.__getattribute__("withdrawFunds")(dest_address).build_transaction({ 'from': user.eth_address,
                                                                                                        'nonce': contract._w3.eth.get_transaction_count(
            user.eth_address) })
    signed_tx = contract._w3.eth.account.sign_transaction(
        tx, private_key=user.private_key)
    tx_hash = contract._w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    return contract._w3.eth.wait_for_transaction_receipt(tx_hash)        

def close_one_side(contract, user: HasEthAccount, channel_state: ChannelStateMessage) -> bool:
    try:
        contract.transact(user, "oneSidedClose", (channel_state.balance1, channel_state.balance2, channel_state.serial_number, 
                                            channel_state.sig[0], channel_state.sig[1], channel_state.sig[2]))
        return True
    except:
        return False

def appeal_closure(contract, user: HasEthAccount, channel_state: ChannelStateMessage) -> bool:
    try:
        contract.transact(user, "appealClosure", (channel_state.balance1, channel_state.balance2, channel_state.serial_number, 
                                            channel_state.sig[0], channel_state.sig[1], channel_state.sig[2]))
        return True
    except:
        return False

