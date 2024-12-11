from .utils import BlockHash, PublicKey, TxID
from .transaction import Transaction
from typing import List, Optional
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
        return self.transactions.copy()

    def get_prev_block_hash(self) -> BlockHash:
        """Gets the hash of the previous block in the chain"""
        return self.prev_block_hash

    def calc_block_hash(self):
        transactions_data = b"|".join(
            (transaction.input if transaction.input is not None else b"") + b";" + transaction.output + b";" + transaction.signature for transaction in self.transactions
        )
        return hashlib.sha256(transactions_data + self.get_prev_block_hash()).digest()
