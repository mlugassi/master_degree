from .utils import BlockHash
from .transaction import Transaction
from typing import List


class Block:
    """This class represents a block."""

    # implement __init__ as you see fit.

    def get_block_hash(self) -> BlockHash:
        """Gets the hash of this block. 
        This function is used by the tests. Make sure to compute the result from the data in the block every time 
        and not to cache the result"""
        raise NotImplementedError()

    def get_transactions(self) -> List[Transaction]:
        """
        returns the list of transactions in this block.
        """
        raise NotImplementedError()

    def get_prev_block_hash(self) -> BlockHash:
        """Gets the hash of the previous block"""
        raise NotImplementedError()
