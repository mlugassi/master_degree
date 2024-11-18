from .utils import *
from .transaction import Transaction
from .bank import Bank
from typing import Optional


class Wallet:
    def __init__(self) -> None:
        """This function generates a new wallet with a new private key."""
        # TODO: Need to generate the keys below
        self.private_key: PrivateKey = None
        self.public_key: PublicKey = None
        self.balance: int = 0
        self.last_updated_blockhash: BlockHash = None


    def update(self, bank: Bank) -> None:
        '''
        This function updates the balance allocated to this wallet by querying the bank.
        Don't read all of the bank's utxo, but rather process the blocks since the last update one at a time.
        For this exercise, there is no need to validate all transactions in the block.
        '''
        latest_blockhash = bank.get_latest_hash()
        curr_blockhash = latest_blockhash

        while(curr_blockhash != self.last_updated_blockhash):
            pass
            curr_blockhash = bank.get_block(curr_blockhash).get_prev_block_hash()
        
        self.last_updated_blockhash = latest_blockhash


        raise NotImplementedError()

    def create_transaction(self, target: PublicKey) -> Optional[Transaction]:
        """
        This function returns a signed transaction that moves an unspent coin to the target.
        It chooses the coin based on the unspent coins that this wallet had since the last update.
        If the wallet already spent a specific coin, but that transaction wasn't confirmed by the
        bank just yet (it still wasn't included in a block) then the wallet should'nt spend it again
        until unfreeze_all() is called. The method returns None if there are no unspent outputs that can be used.
        """
        raise NotImplementedError()

    def unfreeze_all(self) -> None:
        """
        Allows the wallet to try to re-spend outputs that it created transactions for (unless these outputs made it into the blockchain).
        """
        raise NotImplementedError()

    def get_balance(self) -> int:
        """
        This function returns the number of coins that this wallet has.
        It will return the balance according to information gained when update() was last called.
        Coins that the wallet owned and sent away will still be considered as part of the balance until the spending
        transaction is in the blockchain.
        """
        return self.balance

    def get_address(self) -> PublicKey:
        """
        This function returns the public address of this wallet (see the utils module for generating keys).
        """
        return self.public_key