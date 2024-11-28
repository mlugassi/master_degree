import pytest
import sys
 
# sys.path.insert(1,r"C:\Users\rzanzuri\OneDrive - Intel Corporation\Desktop\תואר שני\master_degree\First Year\Introduction to Cryptocurrencies")
# sys.path.insert(1,r"C:\Users\mlugassi\OneDrive - Intel Corporation\Documents\Private\Master Degree\First Year\Introduction to Cryptocurrencies")
from ex1 import Transaction, Wallet, Bank


@pytest.fixture
def bank() -> Bank:
    return Bank()


@pytest.fixture
def bank2() -> Bank:
    return Bank()


@pytest.fixture
def alice() -> Wallet:
    return Wallet()


@pytest.fixture
def alice_coin(bank: Bank, alice: Wallet) -> Transaction:
    bank.create_money(alice.get_address())
    bank.end_day()
    alice.update(bank)
    return bank.get_utxo()[0]


@pytest.fixture
def bob() -> Wallet:
    return Wallet()


@pytest.fixture
def charlie() -> Wallet:
    return Wallet()
