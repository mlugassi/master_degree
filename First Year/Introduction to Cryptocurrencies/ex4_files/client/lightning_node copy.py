from typing import Dict, Optional, List, Any
from client.utils import ChannelStateMessage, EthereumAddress, IPAddress, PrivateKey, Signature
from hexbytes import HexBytes
from eth_typing import HexAddress, HexStr
from web3 import Web3
from client.network import Network
from client.node import Node

class LightningNode(Node):
    """Represents a payment channel node that can support several payment channels."""
    
    def __init__(self, private_key: PrivateKey, eth_address: EthereumAddress, networking_interface: Network, ip: IPAddress, w3: Web3, contract_bytecode: str, contract_abi: Dict[str, Any]) -> None:
        """Creates a new node that uses the given ethereum account (private key and address),
        communicates on the given network and has the provided ip address. 
        It communicates with the blockchain via the supplied Web3 object.
        It is also supplied with the bytecode and ABI of the Channel contract that it will deploy.
        All values are assumed to be legal."""
        
        self.private_key = private_key
        self.eth_address = eth_address
        self.networking_interface = networking_interface
        self.ip_address = ip
        self.w3 = w3
        self.contract_bytecode = contract_bytecode
        self.contract_abi = contract_abi
        self.channels = {}

    def get_list_of_channels(self) -> List[EthereumAddress]:
        """Returns a list of channels managed by this node."""
        return [address for address, channel in self.channels.items() if channel['status'] != 'closed']

    def establish_channel(self, other_party_eth_address: EthereumAddress, other_party_ip_address: IPAddress, amount_in_wei: int) -> EthereumAddress:
        """Creates a new channel that connects this node and the other party, funded by this node."""
        if amount_in_wei <= 0:
            raise ValueError("Amount must be positive.")
        
        # Checking balance (for simplicity, using some placeholder check)
        balance = self.w3.eth.get_balance(self.eth_address)
        if balance < amount_in_wei:
            raise ValueError("Insufficient funds.")
        
        # Deploy contract
        contract = self.w3.eth.contract(abi=self.contract_abi, bytecode=self.contract_bytecode)
        tx = contract.constructor(other_party_eth_address, amount_in_wei, amount_in_wei).buildTransaction({
            'from': self.eth_address,
            'nonce': self.w3.eth.getTransactionCount(self.eth_address),
            'gas': 2000000,
            'gasPrice': self.w3.toWei('20', 'gwei')
        })
        
        signed_tx = self.w3.eth.account.signTransaction(tx, self.private_key)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
        
        channel_address = tx_receipt.contractAddress
        
        # Store the channel
        self.channels[channel_address] = {
            'peer': other_party_eth_address,
            'balance': amount_in_wei,
            'status': 'open',
            'ip_address': other_party_ip_address
        }

        return EthereumAddress(channel_address)

    @property
    def eth_address(self) -> EthereumAddress:
        """Returns the ethereum address of this node."""
        return self.eth_address

    @property
    def ip_address(self) -> IPAddress:
        return self.ip_address

    @property
    def private_key(self) -> PrivateKey:
        """Returns the private key of this node."""
        return self.private_key

    def send(self, channel_address: EthereumAddress, amount_in_wei: int) -> None:
        """Sends money in one of the open channels this node is participating in."""
        channel = self.channels.get(channel_address)
        if not channel:
            raise ValueError("Channel not found.")
        
        if channel['status'] == 'closed':
            raise ValueError("Channel is already closed.")
        
        if amount_in_wei <= 0:
            raise ValueError("Amount must be positive.")

        # Update balances (simplified)
        if self.eth_address == channel['peer']:
            channel['balance'] -= amount_in_wei
        else:
            channel['balance'] += amount_in_wei
        
        # Send funds through the channel contract (simplified)
        contract = self.w3.eth.contract(address=channel_address, abi=self.contract_abi)
        tx = contract.functions.send(channel['peer'], amount_in_wei).buildTransaction({
            'from': self.eth_address,
            'nonce': self.w3.eth.getTransactionCount(self.eth_address),
            'gas': 2000000,
            'gasPrice': self.w3.toWei('20', 'gwei')
        })
        signed_tx = self.w3.eth.account.signTransaction(tx, self.private_key)
        tx_hash = self.w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        self.w3.eth.waitForTransactionReceipt(tx_hash)

    def get_current_channel_state(self, channel_address: EthereumAddress) -> ChannelStateMessage:
        """Gets the latest state of the channel."""
        channel = self.channels.get(channel_address)
        if not channel:
            raise ValueError("Channel not found.")
        
        return ChannelStateMessage(
            EthereumAddress(channel['peer']),
            channel['balance'],
            0,  # Placeholder for amount2
            0,  # Placeholder for some other data
            Signature((0, "", ""))  # Placeholder for signature
        )

    def close_channel(self, channel_address: EthereumAddress, channel_state: Optional[ChannelStateMessage] = None) -> bool:
        """Closes the channel."""
        channel = self.channels.get(channel_address)
        if not channel:
            raise ValueError("Channel not found.")
        
        if channel['status'] == 'closed':
            raise ValueError("Channel already closed.")
        
        channel['status'] = 'closed'
        return True

    def appeal_closed_chan(self, contract_address: EthereumAddress) -> bool:
        """Checks if the channel at the given address needs to be appealed."""
        channel = self.channels.get(contract_address)
        if not channel or channel['status'] != 'closed':
            return False
        
        # Simulate appeal
        return True

    def withdraw_funds(self, contract_address: EthereumAddress) -> int:
        """Withdraws funds from a channel after the appeal period."""
        channel = self.channels.get(contract_address)
        if not channel:
            raise ValueError("Channel not found.")
        
        if channel['status'] != 'closed':
            raise ValueError("Channel is not closed.")
        
        # For simplicity, withdrawing all balance
        withdrawn_amount = channel['balance']
        channel['balance'] = 0
        self.channels[contract_address] = channel
        
        return withdrawn_amount

    def notify_of_channel(self, contract_address: EthereumAddress, other_party_ip_address: IPAddress) -> None:
        """Notifies this node of a new channel."""
        # Check if this node is part of the channel
        if contract_address in self.channels:
            return
        
        self.channels[contract_address] = {
            'peer': None,  # Will be set later when state is synced
            'balance': 0,  # Will be set later
            'status': 'open',
            'ip_address': other_party_ip_address
        }

    def ack_transfer(self, msg: ChannelStateMessage) -> None:
        """Receives a confirmation about a transfer."""
        channel_address = msg.sender
        channel = self.channels.get(channel_address)
        if not channel:
            return
        
        # Simulate checking and accepting the transfer
        if msg.serial_number > 0:
            # Placeholder validation
            channel['balance'] += msg.amount
            self.channels[channel_address] = channel

    def receive_funds(self, state_msg: ChannelStateMessage) -> None:
        """Receives funds and validates the state."""
        channel_address = state_msg.sender
        channel = self.channels.get(channel_address)
        if not channel:
            return
        
        if state_msg.serial_number > 0:
            # Validate transfer
            if state_msg.amount > 0:
                channel['balance'] += state_msg.amount
                self.channels[channel_address] = channel
