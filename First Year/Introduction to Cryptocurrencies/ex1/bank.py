from .utils import GENESIS_BLOCK_PREV, BlockHash, PublicKey, TxID, verify
from .transaction import Transaction
from .block import Block
from typing import List, Optional
import secrets
import copy


class Bank:
    def __init__(self) -> None:
        """Creates a bank with an empty blockchain and an empty mempool."""
        self._blockchain: List[Block] = list()
        self._mempool: List[Transaction] = list()
        self._utxo: List[Transaction] = list()

    def find_transaction(self, txid: Optional[TxID]):
        if txid is not None:
            for block in self._blockchain:
                transaction = block.find_transaction(txid)
                if transaction is not None:
                    return transaction
        return None
    
    def add_transaction_to_mempool(self, transaction: Transaction) -> bool:
        """
        This function inserts the given transaction to the mempool.
        It will return False if one of the following conditions hold:
        (i) the transaction is invalid (the signature fails)
        (ii) the source doesn't have the coin that he tries to spend
        (iii) there is contradicting tx in the mempool.
        (iv) there is no input (i.e., this is an attempt to create money from nothing)
        """
        if transaction is None:
            return False    
        
        input_transaction = self.find_transaction(transaction._input)
        if input_transaction is None: # (iv)
            return False        
        
        if transaction._input not in [txn.get_txid() for txn in self._utxo]: # (ii)
            return False

        if not verify((transaction._input + transaction._output), transaction._signature, input_transaction._output): # (i) # type: ignore
            return False

        for mempool_transaction in self.get_mempool():  # (iii)
            if mempool_transaction._input == transaction._input:
                return False
        self._mempool.append(transaction)
        return True

    def end_day(self, limit: int = 10) -> BlockHash:
        """
        This function tells the bank that the day ended,
        and that the first `limit` transactions in the mempool should be committed to the blockchain.
        If there are fewer than 'limit' transactions in the mempool, a smaller block is created.
        If there are no transactions, an empty block is created. The hash of the block is returned.
        """
        block_transactions: List[Transaction] = list()

        for i, transaction in enumerate(self.get_mempool()[:]):
            if i < limit:
                block_transactions.append(transaction)
                self._mempool.remove(transaction)
                input_transaction = self.find_transaction(transaction._input)
                if input_transaction is not None:
                    self._utxo.remove(input_transaction)
                self._utxo.append(transaction)
            else:
                break
        self._blockchain.append(Block(block_transactions, self.get_latest_hash()))

    def get_block(self, block_hash: BlockHash) -> Block:
        """
        This function returns a block object given its hash. If the block doesnt exist, an exception is thrown..
        """
        for block in self._blockchain:
            if block.get_block_hash() == block_hash:
                return copy.deepcopy(block)
        raise Exception(f"Block hash {block_hash} doen't exists in the blockchain")

    def get_latest_hash(self) -> BlockHash:
        """
        This function returns the hash of the last Block that was created by the bank.
        """
        if len(self._blockchain) == 0:
            return GENESIS_BLOCK_PREV
        return self._blockchain[-1].get_block_hash()

    def get_mempool(self) -> List[Transaction]:
        """
        This function returns the list of transactions that didn't enter any block yet.
        """
        return self._mempool.copy()

    def get_utxo(self) -> List[Transaction]:
        """
        This function returns the list of unspent transactions.
        """
        return self._utxo.copy()

    def create_money(self, target: PublicKey) -> None:
        """
        This function inserts a transaction into the mempool that creates a single coin out of thin air. Instead of a signature,
        this transaction includes a random string of 48 bytes (so that every two creation transactions are different).
        This function is a secret function that only the bank can use (currently for tests, and will make sense in a later exercise).
        """
        new_transaction = Transaction(target, None, secrets.token_bytes(48))
        self._mempool.append(new_transaction)
