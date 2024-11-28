from .utils import BlockHash, PublicKey
from .transaction import Transaction
from typing import List
import hashlib

class Block:
    def __init__(self, transactions: List[Transaction], prev_block_hash: BlockHash):
        self.transactions: List[Transaction] = transactions
        self.prev_block_hash: BlockHash = prev_block_hash
        self.block_hash: BlockHash = self.calc_block_hash()

    def get_block_hash(self) -> BlockHash:
        """returns hash of this block"""
        return self.block_hash

    def get_transactions(self) -> List[Transaction]:
        """returns the list of transactions in this block."""
        return self.transactions

    def get_prev_block_hash(self) -> BlockHash:
        """Gets the hash of the previous block in the chain"""
        return self.prev_block_hash

    def calc_block_hash(self):
        transactions_data = b"|".join(
            transaction.input + b";" + transaction.output for transaction in self.transactions
        )
    
        return hashlib.sha256(transactions_data).digest()
    
    # TODO maybe that func should be removed
    def get_transactions_by_address(self, address: PublicKey) -> List[Transaction]:
        return [tran for tran in self.transactions if tran.output == address]
