from .utils import BlockHash
from .transaction import Transaction
from typing import List
from hashlib import sha256


class Block:
    """This class represents a block."""

    # implement __init__ as you see fit.
    def __init__(self, transactions: List[Transaction], prev_block_hash: BlockHash):
        self.transactions: List[Transaction] = transactions
        self.prev_block_hash: BlockHash = prev_block_hash
        self.block_hash: BlockHash = self.calc_block_hash()

    def get_block_hash(self) -> BlockHash:
        """Gets the hash of this block. 
        This function is used by the tests. Make sure to compute the result from the data in the block every time 
        and not to cache the result"""
        return self.calc_block_hash()

    def get_transactions(self) -> List[Transaction]:
        """
        returns the list of transactions in this block.
        """
        return self.transactions

    def get_prev_block_hash(self) -> BlockHash:
        """Gets the hash of the previous block"""
        return self.prev_block_hash

    def calc_block_hash(self):
        transactions_data = b"|".join(
            (transaction.input if transaction.input is not None else b"") + b";" + transaction.output + b";" + transaction.signature for transaction in self.transactions
        )
        return sha256(transactions_data + self.get_prev_block_hash()).digest()

