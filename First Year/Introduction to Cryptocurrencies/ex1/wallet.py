from .utils import *
from .transaction import Transaction
from .bank import Bank
from typing import List, Optional


class Wallet:
    def __init__(self) -> None:
        """This function generates a new wallet with a new private key."""
        self.private_key, self.public_key = gen_keys()
        self.unspent_transaction: List[Transaction] = list()
        self.pending_transaction: List[Transaction] = list()
        self.last_updated_blockhash: BlockHash = GENESIS_BLOCK_PREV

    def update(self, bank: Bank) -> None:
        """
        This function updates the balance allocated to this wallet by querying the bank.
        Don't read all of the bank's utxo, but rather process the blocks since the last update one at a time.
        For this exercise, there is no need to validate all transactions in the block.
        """
        latest_blockhash = bank.get_latest_hash()
        curr_blockhash = latest_blockhash

        while curr_blockhash != self.last_updated_blockhash:
            curr_block = bank.get_block(curr_blockhash)
            transactions = curr_block.get_transactions()
            for transaction in transactions:
                if transaction.output == self.get_address():
                    self.unspent_transaction.append(transaction)
                elif transaction in self.pending_transaction:
                    self.pending_transaction.remove(transaction)
            curr_blockhash = curr_block.get_prev_block_hash()

        self.last_updated_blockhash = latest_blockhash

    def create_transaction(self, target: PublicKey) -> Optional[Transaction]:
        """
        This function returns a signed transaction that moves an unspent coin to the target.
        It chooses the coin based on the unspent coins that this wallet had since the last update.
        If the wallet already spent a specific coin, but that transaction wasn't confirmed by the
        bank just yet (it still wasn't included in a block) then the wallet should'nt spend it again
        until unfreeze_all() is called. The method returns None if there are no unspent outputs that can be used.
        """
        if not self.unspent_transaction:
            return None
        
        chosen_transction = self.unspent_transaction.pop()
        self.pending_transaction.append(chosen_transction)
        if chosen_transction.input is None:
            signature = sign(chosen_transction.output, self.private_key)
        else:
            signature = sign((chosen_transction.input + chosen_transction.output), self.private_key)
        return Transaction(output=target, input=chosen_transction.get_txid(), signature=signature)

    def unfreeze_all(self) -> None:
        """
        Allows the wallet to try to re-spend outputs that it
        created transactions for (unless these outputs made it into the blockchain).
        """
        self.unspent_transaction += self.pending_transaction
        self.pending_transaction.clear()

    def get_balance(self) -> int:
        """
        This function returns the number of coins that this wallet has.
        It will return the balance according to information gained when update() was last called.
        Coins that the wallet owned and sent away will still be considered as part of the balance until the spending
        transaction is in the blockchain.
        """
        return len(self.unspent_transaction) + len(self.pending_transaction)

    def get_address(self) -> PublicKey:
        """
        This function returns the public address of this wallet (see the utils module for generating keys).
        """
        return self.public_key
