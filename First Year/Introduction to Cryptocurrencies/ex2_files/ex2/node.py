from .utils import *
from .block import Block
from .transaction import Transaction
from typing import Set, Optional, List
from secrets import token_bytes
from hashlib import sha256

class Node:
    def __init__(self) -> None:
        """Creates a new node with an empty mempool and no connections to others.
        Blocks mined by this node will reward the miner with a single new coin,
        created out of thin air and associated with the mining reward address"""
        self.blockchain: List[Block] = list()
        self.mempool: List[Transaction] = list()
        self.utxo: List[Transaction] = list()
        self.private_key, self.public_key = gen_keys()
        self.unspent_transaction: List[Transaction] = list()
        self.pending_transaction: List[Transaction] = list()
        self.connected_nodes: Set['Node'] = set()

    def connect(self, other: 'Node') -> None:
        """connects this node to another node for block and transaction updates.
        Connections are bi-directional, so the other node is connected to this one as well.
        Raises an exception if asked to connect to itself.
        The connection itself does not trigger updates about the mempool,
        but nodes instantly notify of their latest block to each other (see notify_of_block)"""
        if type(other) is not Node:
            return None
        if other.get_address() == self.get_address():
            raise Exception("You are trying to connect to yourself")
        if other.get_address() not in [n.get_address() for n in self.connected_nodes]:
            self.connected_nodes.add(other)
            other.connect(self)
            other.notify_of_block(self.get_latest_hash(), self)

    def disconnect_from(self, other: 'Node') -> None:
        """Disconnects this node from the other node. If the two were not connected, then nothing happens"""
        if type(other) is not Node:
            return None
        for n in self.connected_nodes:
            if n.get_address() == other.get_address():
                self.connected_nodes.remove(n)
                other.disconnect_from(self)
                break

    def get_connections(self) -> Set['Node']:
        """Returns a set containing the connections of this node."""
        return self.connected_nodes

    def add_transaction_to_mempool(self, transaction: Transaction) -> bool:
        """
        This function inserts the given transaction to the mempool.
        It will return False if any of the following conditions hold:
        (i) the transaction is invalid (the signature fails)
        (ii) the source doesn't have the coin that it tries to spend
        (iii) there is contradicting tx in the mempool.

        If the transaction is added successfully, then it is also sent to neighboring nodes.
        """
        if transaction is None or transaction.input is None:
            return False    

        input_transaction = self.find_transaction_in_utxo(transaction.input) # (ii)
        if input_transaction is None:
            return False        
            
        if not verify((transaction.input + transaction.output), transaction.signature, input_transaction.output): # (i)
            return False

        for mempool_transaction in self.get_mempool():  # (iii)
            if mempool_transaction.input == transaction.input:
                return False
            
        self.mempool.append(transaction)
        # Notify to the neighbores
        for neighbor in self.connected_nodes:
            neighbor.add_transaction_to_mempool(transaction)

        return True
    

    def notify_of_block(self, block_hash: BlockHash, sender: 'Node') -> None:
        """This method is used by a node's connection to inform it that it has learned of a
        new block (or created a new block). If the block is unknown to the current Node, The block is requested.
        We assume the sender of the message is specified, so that the node can choose to request this block if
        it wishes to do so.
        (if it is part of a longer unknown chain, these blocks are requested as well, until reaching a known block).
        do, they are processed and checked for validity (check all signatures, hashes,
        block size , etc).
        If the block is on the longest chain, the mempool and utxo change accordingly.
        If the block is indeed the top of the longest chain,
        a notification of this block is sent to the neighboring nodes of this node.
        (no need to notify of previous blocks -- the nodes will fetch them if needed)

        A reorg may be triggered by this block's introduction. In this case the utxo is rolled back to the split point,
        and then rolled forward along the new branch.
        the mempool is similarly emptied of transactions that cannot be executed now.
        transactions that were rolled back and can still be executed are re-introduced into the mempool if they do
        not conflict.
        """
        curr_block_hash = block_hash
        unkonwn_blocks_to_me: List[Block] = list()
        unkonwn_blocks_to_hashes: List[BlockHash] = list()
        num_of_unknown_blocks_to_sender = None
       
        while num_of_unknown_blocks_to_sender is None:
            if curr_block_hash in unkonwn_blocks_to_hashes: # Checking if sender blockchain has infinity loop
                return None
            for i, my_block in enumerate(self.blockchain[::-1]):
                if my_block.get_block_hash() == curr_block_hash:
                    num_of_unknown_blocks_to_sender = i
                    break
            else:
                try:
                    sender_block = sender.get_block(curr_block_hash)
                except: # if we failed here it's mean that the curr_block_hash isn't point on real block
                    return None
                unkonwn_blocks_to_me.insert(0, sender_block)
                unkonwn_blocks_to_hashes.insert(0, curr_block_hash)
                if sender_block.get_prev_block_hash() != GENESIS_BLOCK_PREV:
                    curr_block_hash = sender_block.get_prev_block_hash()
                else:
                    num_of_unknown_blocks_to_sender = len(self.blockchain)

        forked_blockchain = self.blockchain[:len(self.blockchain) - num_of_unknown_blocks_to_sender].copy()
        for i, (unknown_block, unkonwn_block_hash) in enumerate(zip(unkonwn_blocks_to_me, unkonwn_blocks_to_hashes)):
            if not self.validate_block(unknown_block, unkonwn_block_hash, forked_blockchain + unkonwn_blocks_to_me[:i + 1]):
                unkonwn_blocks_to_me = unkonwn_blocks_to_me[:i]
                break
                
        if num_of_unknown_blocks_to_sender >= len(unkonwn_blocks_to_me):
            return None            

        # Removing my blocks that are't in the longest blockchain
        removed_from_mempool = list()
        for i in range(num_of_unknown_blocks_to_sender):
            block = self.blockchain.pop()
            for removed_tx in block.get_transactions():
                self.remove_transaction_from_utxo(removed_tx)
                # Check if in mempool there is a transaction that is basing on the removed transaction 
                # (and save it in different list since it can be added if the removed transaction 
                # will be added from the sender's new blocks)
                for tx in self.mempool.copy():
                    if tx.input == removed_tx.get_txid():
                        removed_from_mempool.append(tx)
                        self.mempool.remove(tx)
                        break

        for block in unkonwn_blocks_to_me:
            self.blockchain.append(block)
            for tx in block.get_transactions():
                self.add_transaction_to_utxo(tx)

        for removed_tx in removed_from_mempool:
            for tx in self.get_utxo():
                if removed_tx.input == tx.get_txid():
                    self.mempool.append(removed_tx)
                    break
        
        for tx in self.mempool.copy():
            for utxo_tx in self.get_utxo():
                if tx.input == utxo_tx.get_txid():
                    break
            else:
                self.mempool.remove(tx)

        for neighbor in self.connected_nodes:
             neighbor.notify_of_block(unkonwn_blocks_to_me[-1].get_block_hash(), self)

    def mine_block(self) -> BlockHash:
        """"
        This function allows the node to create a single block.
        The block should contain BLOCK_SIZE transactions (unless there aren't enough in the mempool). Of these,
        BLOCK_SIZE-1 transactions come from the mempool and one addtional transaction will be included that creates
        money and adds it to the address of this miner.
        Money creation transactions have None as their input, and instead of a signature, contain 48 random bytes.
        If a new block is created, all connections of this node are notified by calling their notify_of_block() method.
        The method returns the new block hash (or None if there was no block)
        """
        transactions: List[Transaction] = list()

        transactions.append(Transaction(self.get_address(), None, token_bytes(64)))
        
        for tx in self.get_mempool():
            if len(transactions) < BLOCK_SIZE:
                transactions.append(tx)
        
        for tx in transactions:
            self.add_transaction_to_utxo(tx)
        new_block = Block(self.get_latest_hash(), transactions)
        self.blockchain.append(new_block)
        for node in self.connected_nodes:
            node.notify_of_block(new_block.get_block_hash(), self)
        
        return new_block.get_block_hash()

    def get_block(self, block_hash: BlockHash) -> Block:
        """
        This function returns a block object given its hash.
        If the block doesn't exist, a ValueError is raised.
        """
        for block in self.blockchain:
            if block.get_block_hash() == block_hash:
                return block
            
        raise ValueError(f"Block hash {block_hash} doen't exists in the blockchain.")

    def get_latest_hash(self) -> BlockHash:
        """
        This function returns the last block hash known to this node (the tip of its current chain).
        """

        if len(self.blockchain) == 0:
            return GENESIS_BLOCK_PREV
        return self.blockchain[-1].get_block_hash()

    def get_mempool(self) -> List[Transaction]:
        """
        This function returns the list of transactions that didn't enter any block yet.
        """
        return self.mempool

    def get_utxo(self) -> List[Transaction]:
        """
        This function returns the list of unspent transactions.
        """
        return self.utxo

    # ------------ Formerly wallet methods: -----------------------

    def create_transaction(self, target: PublicKey) -> Optional[Transaction]:
        """
        This function returns a signed transaction that moves an unspent coin to the target.
        It chooses the coin based on the unspent coins that this node has.
        If the node already tried to spend a specific coin, and such a transaction exists in its mempool,
        but it did not yet get into the blockchain then it should'nt try to spend it again (until clear_mempool() is
        called -- which will wipe the mempool and thus allow to attempt these re-spends).
        The method returns None if there are no outputs that have not been spent already.

        The transaction is added to the mempool (and as a result is also published to neighboring nodes)
        """
        if not self.unspent_transaction or type(target) is not bytes:
            return None
        
        chosen_transction = self.unspent_transaction.pop()
        self.pending_transaction.append(chosen_transction)
        signature = sign(chosen_transction.get_txid() + target, self.private_key)
        new_tx = Transaction(output=target, tx_input=chosen_transction.get_txid(), signature=signature)
        if self.add_transaction_to_mempool(new_tx):
            return new_tx
        return None
        
    def clear_mempool(self) -> None:
        """
        Clears the mempool of this node. All transactions waiting to be entered into the next block are gone.
        """

        self.unspent_transaction.clear()
        self.pending_transaction.clear()
        self.mempool.clear()

        for tx in self.get_utxo():
            if tx.output == self.get_address():
                self.unspent_transaction.append(tx)
   
    def get_balance(self) -> int:
        """
        This function returns the number of coins that this node owns according to its view of the blockchain.
        Coins that the node owned and sent away will still be considered as part of the balance until the spending
        transaction is in the blockchain.
        """
        return len(self.unspent_transaction) + len(self.pending_transaction)

    def get_address(self) -> PublicKey:
        """
        This function returns the public address of this node (its public key).
        """
        return self.public_key

    def find_transaction_in_utxo(self, txid: Optional[TxID]):
        for tx in self.utxo:
            if tx.get_txid() == txid:
                return tx
        return None

    def add_transaction_to_utxo(self, added_tx: Transaction) -> None:
        if added_tx.input is not None and added_tx in self.mempool:
            self.mempool.remove(added_tx)
        self.utxo.append(added_tx)
        if added_tx.output == self.get_address():
            self.unspent_transaction.append(added_tx)
        
        if added_tx.input is not None:
            input_tx = find_transaction(self.blockchain, added_tx.input)[0]
            self.utxo.remove(input_tx)
            if input_tx in self.unspent_transaction:
                self.unspent_transaction.remove(input_tx)
            elif input_tx in self.pending_transaction:
                self.pending_transaction.remove(input_tx)                
    
    def remove_transaction_from_utxo(self, removed_tx: Transaction) -> None:
        self.utxo.remove(removed_tx)
        if removed_tx.output == self.get_address():
            if removed_tx in self.unspent_transaction:
                self.unspent_transaction.remove(removed_tx)
            elif removed_tx in self.pending_transaction:
                self.pending_transaction.remove(removed_tx)
            else:
                raise Exception("Transaction must be in pending or unsent lists")            
        
        if removed_tx.input == None:
            return None
        self.mempool.append(removed_tx) # TODO check if we need add removed transction to mempool after removing from the utxo (and block)
        input_tx = find_transaction(self.blockchain, removed_tx.input)[0]
        self.utxo.append(input_tx)
        if input_tx.output == self.get_address():
            self.unspent_transaction.append(input_tx)
    
    def calc_block_hash(self, transactions: List[Transaction], prev_block_hash: BlockHash):
        transactions_data = b"|".join(
            (transaction.input if transaction.input is not None else b"") + b";" + transaction.output + b";" + transaction.signature for transaction in transactions
        )
        return sha256(transactions_data + prev_block_hash).digest()
        
    def validate_block(self, block: Block, block_hash: BlockHash, blockchain: List[Block]) -> bool:
        # num of transaction in block
        if len(block.get_transactions()) > BLOCK_SIZE:
            return False
        # any tx in tx of block
        num_of_mined_transctions = 0
        for tx in block.get_transactions():
            if not tx or tx.output is None:
                return False
            if tx.input is None:
                num_of_mined_transctions += 1
            else:
                input_tx = find_transaction(blockchain, tx.input)
                if len(input_tx) != 1:
                    return False
                if not verify((tx.input + tx.output), tx.signature, input_tx[0].output):
                    return False
        if num_of_mined_transctions != 1:
            return False
        all_inputs = [tx.input for block in blockchain for tx in block.get_transactions() if tx.input != None]
        if len(all_inputs) > len(set(all_inputs)):
            return False
        all_sigs = [tx.signature for block in blockchain for tx in block.get_transactions()]
        if len(all_sigs) > len(set(all_sigs)):
            return False
        
        if self.calc_block_hash(block.get_transactions(), block.get_prev_block_hash()) != block_hash:
            return False
        
        return True

def find_transaction(blockchain: List[Block], txid: TxID) -> List[Transaction]:
    found_transactions: List[Transaction] = list()
    for block in blockchain:
        tx = block.find_transaction(txid)
        if tx:
            found_transactions.append(tx)
    return found_transactions

"""
Importing this file should NOT execute code. It should only create definitions for the objects above.
Write any tests you have in a different file.
You may add additional methods, classes and files but be sure no to change the signatures of methods
included in this template.
"""