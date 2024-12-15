import sys
from ex1 import *

def test_block(bank: Bank, alice_coin: Transaction) -> None:
    hash1 = bank.get_latest_hash()
    block = bank.get_block(hash1)
    assert len(block.get_transactions()) == 1
    assert block.get_prev_block_hash() == GENESIS_BLOCK_PREV

    bank.end_day()

    hash2 = bank.get_latest_hash()
    block2 = bank.get_block(hash2)
    assert len(block2.get_transactions()) == 0
    assert block2.get_prev_block_hash() == hash1


def test_create_money_happy_flow(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    alice.update(bank)
    bob.update(bank)
    assert alice.get_balance() == 1
    assert bob.get_balance() == 0
    utxo = bank.get_utxo()
    assert len(utxo) == 1
    assert utxo[0].output == alice.get_address()


def test_transaction_happy_flow(bank: Bank, alice: Wallet, bob: Wallet,
                                alice_coin: Transaction) -> None:
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    assert bank.add_transaction_to_mempool(tx)
    assert bank.get_mempool() == [tx]
    bank.end_day(limit=1)
    alice.update(bank)
    bob.update(bank)
    assert alice.get_balance() == 0
    assert bob.get_balance() == 1
    assert not bank.get_mempool()
    assert bank.get_utxo()[0].output == bob.get_address()
    assert tx.get_txid() == bank.get_block(bank.get_latest_hash()).get_transactions()[0].get_txid()


def test_re_transmit_the_same_transaction(bank: Bank, alice: Wallet, bob: Wallet,
                                          alice_coin: Transaction) -> None:
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    assert bank.add_transaction_to_mempool(tx)
    assert not bank.add_transaction_to_mempool(tx)
    assert bank.get_mempool() == [tx]


def test_spend_coin_not_mine(bank2: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    assert not bank2.add_transaction_to_mempool(tx)
    assert not bank2.get_mempool()


def test_change_output_of_signed_transaction(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet,
                                             alice_coin: Transaction) -> None:
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    tx = Transaction(output=charlie.get_address(),
                     input=tx.input, signature=tx.signature)
    assert not bank.add_transaction_to_mempool(tx)
    assert not bank.get_mempool()
    bank.end_day()
    alice.update(bank)
    bob.update(bank)
    assert alice.get_balance() == 1
    assert bob.get_balance() == 0
    assert charlie.get_balance() == 0


def test_change_coin_of_signed_transaction(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet,
                                           alice_coin: Transaction) -> None:
    # Give Bob two coins
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    bank.add_transaction_to_mempool(tx)
    bank.create_money(bob.get_address())
    bank.end_day()
    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)
    bob_coin1, bob_coin2 = bank.get_utxo()
    # Bob gives a coin to Charlie, and Charlie wants to steal the second one
    tx = bob.create_transaction(charlie.get_address())
    assert tx is not None
    tx2 = Transaction(output=tx.output, input=bob_coin2.get_txid() if tx.input == bob_coin1.get_txid()
                      else bob_coin1.get_txid(), signature=tx.signature)
    assert not bank.add_transaction_to_mempool(tx2)
    assert not bank.get_mempool()
    assert bank.add_transaction_to_mempool(tx)
    assert bank.get_mempool()
    bank.end_day()
    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)
    assert alice.get_balance() == 0
    assert bob.get_balance() == 1
    assert charlie.get_balance() == 1


def test_double_spend_fail(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet, alice_coin: Transaction) -> None:
    tx1 = alice.create_transaction(bob.get_address())
    assert tx1 is not None
    # make alice spend the same coin
    alice.update(bank)
    alice.unfreeze_all()
    tx2 = alice.create_transaction(charlie.get_address())
    assert tx2 is not None  # Alice will try to double spend

    assert bank.add_transaction_to_mempool(tx1)
    assert not bank.add_transaction_to_mempool(tx2)
    bank.end_day(limit=2)
    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)
    assert alice.get_balance() == 0
    assert bob.get_balance() == 1
    assert charlie.get_balance() == 0

def test_send_coin_to_myself(bank: Bank, alice: Wallet, alice_coin: Transaction) -> None:
    tx1 = alice.create_transaction(alice.get_address())
    assert tx1 is not None
    assert alice.get_balance() == 1
    tx1 = alice.create_transaction(alice.get_address())
    assert tx1 is None

    assert alice.get_balance() == 1

    # make alice spend the same coin
    alice.update(bank)
    alice.unfreeze_all()
    assert alice.get_balance() == 1


def test_invalid_signature(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Create a valid transaction and tamper with the signature
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    tampered_tx = Transaction(output=tx.output, input=tx.input, signature=b"tampered_signature")
    assert not bank.add_transaction_to_mempool(tampered_tx)
    assert not bank.get_mempool()


def test_empty_transaction(bank: Bank, alice: Wallet) -> None:
    # Test that a transaction cannot be created without a valid input and output
    tx = Transaction(output=None, input=None, signature=None)
    assert not bank.add_transaction_to_mempool(tx)
    assert not bank.get_mempool()


def test_insufficient_funds(bank: Bank, alice: Wallet, bob: Wallet) -> None:
    # Try to create a transaction from a wallet with insufficient funds
    alice.update(bank)
    tx = alice.create_transaction(bob.get_address())
    assert tx is None
    bank.add_transaction_to_mempool(tx)
    bank.end_day()

    # Alice tries to spend again without sufficient funds
    alice.update(bank)
    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is None  # Transaction shouldn't be created
    assert alice.get_balance() == 0


def test_create_multiple_blocks(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Ensure that multiple blocks are created correctly
    tx1 = alice.create_transaction(bob.get_address())
    assert tx1 is not None
    bank.add_transaction_to_mempool(tx1)
    bank.end_day()
    alice.update(bank)

    tx2 = bob.create_transaction(alice.get_address())
    assert tx2 is None
    bank.add_transaction_to_mempool(tx2)
    bank.end_day()

    bob.update(bank)
    alice.unfreeze_all()
    tx3 = alice.create_transaction(bob.get_address())
    assert tx3 is None
    bank.add_transaction_to_mempool(tx3)
    bank.end_day()

    alice.update(bank)

    assert len(bank.get_utxo()) == 1
    assert bank.get_block(bank.get_block(bank.get_latest_hash()).get_prev_block_hash()) is not None


def test_no_transactions_no_block(bank: Bank) -> None:
    # Test that no block is created if no transactions are in the mempool
    initial_hash = bank.get_latest_hash()
    bank.end_day()
    assert bank.get_latest_hash() is not initial_hash

def test_unfreeze_and_reuse(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Test unfreeze functionality and reuse of coins
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    bank.add_transaction_to_mempool(tx)
    bank.end_day()

    # Try to reuse the coin with unfreezing
    alice.unfreeze_all()
    assert alice.get_balance() == 1
    alice.update(bank)
    assert alice.get_balance() == 0
    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is None
    bank.end_day()
    alice.update(bank)
    bob.update(bank)

    # Unfreeze and retry
    tx3 = alice.create_transaction(bob.get_address())
    assert tx3 is None
    assert bank.add_transaction_to_mempool(tx3) is False


def test_large_transaction_pool(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet) -> None:
    # Create a large number of transactions to test the mempool and block creation
    for _ in range(100):
        bank.create_money(alice.get_address())

    assert len(bank.get_mempool()) == 100
    bank.end_day(limit=40)  # Process only 40 transactions
    assert len(bank.get_mempool()) == 60  # Remaining transactions
    assert len(bank.get_block(bank.get_latest_hash()).get_transactions()) == 40
    alice.update(bank)
    assert alice.get_balance() == 40
    for _ in range(40):
        tx = alice.create_transaction(bob.get_address())
        assert tx is not None
        bank.add_transaction_to_mempool(tx)
    tx = alice.create_transaction(bob.get_address())
    assert tx is None
    bank.add_transaction_to_mempool(tx)


def test_chain_integrity(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Ensure that the blockchain remains intact after multiple blocks
    hash1 = bank.get_latest_hash()
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    bank.add_transaction_to_mempool(tx)
    bank.end_day()

    hash2 = bank.get_latest_hash()
    assert hash2 != hash1
    block = bank.get_block(hash2)
    assert block.get_prev_block_hash() == hash1

    tx2 = bob.create_transaction(alice.get_address())
    assert tx2 is None
    bob.update(bank)
    tx2 = bob.create_transaction(alice.get_address())
    assert tx2 is not None
    bank.add_transaction_to_mempool(tx2)
    bank.end_day()

    hash3 = bank.get_latest_hash()
    assert hash3 != hash2
    block2 = bank.get_block(hash3)
    assert block2.get_prev_block_hash() == hash2

def test_create_transaction_with_empty_address(bank: Bank, alice: Wallet) -> None:
    # Test if creating a transaction with an empty address fails
    tx = alice.create_transaction("")
    assert tx is None  # Should not create a transaction

def test_add_duplicate_transaction_to_block(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Ensure duplicate transactions aren't added to blocks
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    bank.add_transaction_to_mempool(tx)
    bank.end_day()

    # Try to add the same transaction again
    assert not bank.add_transaction_to_mempool(tx)
    assert len(bank.get_block(bank.get_latest_hash()).get_transactions()) == 1  # Only one copy in the block

def test_invalid_block_access(bank: Bank) -> None:
    # Try accessing a block that doesn't exist
    invalid_hash = "non_existent_hash"
    try:
        block = bank.get_block(invalid_hash)
    except Exception as e:
    # Check the exception message
        assert str(e) == f"Block hash {invalid_hash} doen't exists in the blockchain"

def test_send_transaction_to_invalid_address(bank: Bank, alice: Wallet) -> None:
    # Test if transactions to invalid addresses are blocked
    invalid_address = "invalid_address_123"
    tx = alice.create_transaction(invalid_address)
    assert tx is None  # Transaction shouldn't be created


def test_end_day_with_no_mempool(bank: Bank) -> None:
    # Test if ending a day with an empty mempool creates a new block
    initial_hash = bank.get_latest_hash()
    bank.end_day()
    assert bank.get_latest_hash() != initial_hash  # Hash shouldn't be the same
    assert len(bank.get_block(bank.get_latest_hash()).get_transactions()) == 0

def test_all_utxos_used(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Test behavior when all UTXOs are spent
    tx1 = alice.create_transaction(bob.get_address())
    assert tx1 is not None
    bank.add_transaction_to_mempool(tx1)
    bank.end_day()

    # Alice tries to spend again without any UTXOs
    alice.update(bank)
    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is None  # No UTXOs left for Alice


def test_chain_rollback(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    # Simulate a scenario where chain integrity is broken and rollback is required
    hash1 = bank.get_latest_hash()
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None
    bank.add_transaction_to_mempool(tx)
    bank.end_day()

    # Tamper with the chain
    tampered_block1 = bank.get_block(hash1)
    tampered_block1.prev_block_hash = "tampered_hash"  # Break integrity
    # bank._utxo.clear()
    
    tampered_block2 = bank.get_block(hash1)
    assert tampered_block2.prev_block_hash != tampered_block1.prev_block_hash
    # Test if the bank detects the issue
    # assert not bank.is_chain_valid()  # Assuming you have an integrity check method


def test_multiple_transactions_from_different_wallets(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet) -> None:
    # Test simultaneous transactions from different wallets
    bank.create_money(alice.get_address())
    bank.create_money(bob.get_address())
    bank.create_money(charlie.get_address())
    bank.end_day(limit=2)
    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)
    tx1 = alice.create_transaction(bob.get_address())
    tx2 = bob.create_transaction(charlie.get_address())
    assert tx1 is not None
    assert tx2 is not None
    assert bank.add_transaction_to_mempool(tx1)
    assert bank.add_transaction_to_mempool(tx2)
    assert alice.get_balance() == 1
    assert bob.get_balance() == 1
    assert charlie.get_balance() == 0
    bank.end_day(limit=2)

    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)

    assert alice.get_balance() == 0
    assert bob.get_balance() == 2
    assert charlie.get_balance() == 1

    tx3 = bob.create_transaction(alice.get_address())
    tx3.output = charlie.get_address()
    assert bank.add_transaction_to_mempool(tx3) == False
    import secrets
    new_transaction = Transaction(bob.get_address(), None, secrets.token_bytes(48))
    assert not bank.add_transaction_to_mempool(new_transaction)

    tx4 = charlie.create_transaction(alice.get_address())
    bank.add_transaction_to_mempool(tx4)
    bank.end_day()
    bob.unspent_transaction.append(tx4)
    tx6 = bob.create_transaction(charlie.get_address())
    assert not bank.add_transaction_to_mempool(tx6)
    bank.end_day()
    bob.update(bank)
    charlie.update(bank)


def test_send_two_coins_happy_flow(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    bank.create_money(alice.get_address())
    bank.end_day()
    alice.update(bank)

    tx = alice.create_transaction(bob.get_address())
    assert tx is not None

    bank.add_transaction_to_mempool(tx)

    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is not None


def test_double_spend_next_day(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet, alice_coin: Transaction) -> None:
    tx1 = alice.create_transaction(bob.get_address())
    assert tx1 is not None

    # Make Alice spend the same coin
    bank.end_day()
    alice.update(bank)
    alice.unfreeze_all()

    tx2 = alice.create_transaction(charlie.get_address())
    assert tx2 is not None

    assert bank.add_transaction_to_mempool(tx1)

    bank.end_day()
    assert not bank.add_transaction_to_mempool(tx2)

    bank.end_day()
    alice.update(bank)
    bob.update(bank)
    charlie.update(bank)

    assert alice.get_balance() == 0


def test_re_transmit_similar_txs(bank: Bank, alice: Wallet, bob: Wallet) -> None:
    # If they fail here, then they didn't add randomness to each tx
    # Creating money twice in the same block is important for the next ex.
    bank.create_money(alice.get_address())
    bank.create_money(alice.get_address())
    bank.end_day()

    alice.update(bank)

    tx = alice.create_transaction(bob.get_address())
    assert tx is not None

    assert bank.add_transaction_to_mempool(tx)
    assert not bank.add_transaction_to_mempool(tx)

    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is not None
    assert bank.add_transaction_to_mempool(tx2)


def test_alice_sends_two_coins_to_bob_then_bob_sends_both_to_charlie(bank: Bank, alice: Wallet, bob: Wallet, charlie: Wallet) -> None:
    # If they fail here, then they didn't add randomness to each tx
    # Creating money twice in the same block is important for the next ex.
    bank.create_money(alice.get_address())
    bank.create_money(alice.get_address())
    bank.end_day()

    alice.update(bank)

    tx1 = alice.create_transaction(bob.get_address())
    assert tx1 is not None

    tx2 = alice.create_transaction(bob.get_address())
    assert tx2 is not None

    assert bank.add_transaction_to_mempool(tx1)
    assert bank.add_transaction_to_mempool(tx2)


def test_transaction_happy_flow(bank: Bank, alice: Wallet, bob: Wallet, alice_coin: Transaction) -> None:
    tx = alice.create_transaction(bob.get_address())
    assert tx is not None

    assert bank.add_transaction_to_mempool(tx)
    assert bank.get_mempool() == [tx]

    bank.end_day(limit=1)
    alice.update(bank)
    bob.update(bank)

    assert alice.get_balance() == 0
    assert bob.get_balance() == 1
    assert not bank.get_mempool()
    assert bank.get_utxo()[0].output == bob.get_address()

    assert tx.get_txid() == bank.get_block(bank.get_latest_hash()).get_transactions()[0].get_txid()

