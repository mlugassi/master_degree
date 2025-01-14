import eth_typing
from web3.types import Wei
from web3 import Web3
from logger import logger
import secrets
import pytest
from typing import Any, Tuple
import web3.contract

REVEAL_TIME: int = 5
ONE_ETH = 10**18

ROCK = 1
PAPER = 2
SCISSORS = 3

NO_GAME = 0
MOVE1 = 1
MOVE2 = 2
REVEAL1 = 3
LATE = 4

Account = eth_typing.ChecksumAddress
Accounts = Tuple[Account, ...]
RevertException = web3.exceptions.ContractLogicError

def check_send_money(w3: Web3, from_addr: Account, to_addr: Account, amount: int) -> None:
    """send money from one account to another and check if the transaction is successful"""
    logger.info(f"Sending {amount} wei from {from_addr} to {to_addr}")
    tx_hash = w3.eth.send_transaction({
        'to': to_addr,
        'from': from_addr,
        'value': Wei(amount)
    })
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    logger.info(
        f"Transaction receipt received. Status: {tx_receipt['status']}")
    assert tx_receipt["status"] == 1

def deploy_compiled_contract(w3: Web3, bytecode: str, abi: Any, from_account: Account, *args: Any) -> web3.contract.Contract:
    """Deploy a compiled contract"""
    logger.info("Deploying the contract")
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    ctor = contract.constructor(*args)
    logger.info(f"sending the deployment transaction from {from_account}")
    tx_hash = ctor.transact({'from': from_account})

    logger.info(
        f"Waiting for the transaction receipt of the deployment transaction {tx_hash}")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt["status"] == 1

    logger.info(f"Contract deployed at {tx_receipt['contractAddress']}")
    deployed_contract = w3.eth.contract(
        address=tx_receipt["contractAddress"], abi=abi)

    return deployed_contract

class RPS:
    def __init__(self, w3: Web3, rps_contract: web3.contract.Contract) -> None:
        self.w3 = w3
        self.contract = rps_contract

    @property
    def address(self) -> Account:
        return Account(self.contract.address)

    def get_game_state(self, game_id: int) -> int:
        return int(self.contract.functions.getGameState(game_id).call())

    def withdraw(self, amount: int, from_account: Account) -> None:
        logger.info(
            f"Calling RPS.withdraw to withdraw {amount} from {from_account}")
        tx_hash = self.contract.functions.withdraw(
            amount).transact({'from': from_account})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] == 1

    def make_move(self, game_id: int, bet_ammount: int, move_commit: bytes, from_account: Account) -> None:
        logger.info(
            f"Calling RPS.makeMove with bet {bet_ammount} in game {game_id} from {from_account}")
        tx_hash = self.contract.functions.makeMove(game_id, bet_ammount, move_commit).transact(
            {'from': from_account})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] == 1

    def reveal_move(self, game_id: int, move: int, key: bytes, from_account: Account) -> None:
        logger.info(
            f"Calling RPS.revealMove with move {move} in game {game_id} from {from_account}")
        tx_hash = self.contract.functions.revealMove(
            game_id, move, key).transact({'from': from_account})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] == 1

    def cancel_game(self, game_id: int, from_account: Account) -> None:
        logger.info(
            f"Calling RPS.cancelGame with game id {game_id} from {from_account}")
        tx_hash = self.contract.functions.cancelGame(game_id).transact(
            {'from': from_account})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] == 1

    def reveal_phase_ended(self, game_id: int, from_account: Account) -> None:
        logger.info(
            f"Calling RPS.revealPhaseEnded for game {game_id} from {from_account}")
        tx_hash = self.contract.functions.revealPhaseEnded(game_id).transact(
            {'from': from_account})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        assert tx_receipt["status"] == 1

    def balance_of(self, x: Account) -> int:
        return int(self.contract.functions.balanceOf(x).call())


@ pytest.fixture(scope="module")
def rps(w3: Web3, accounts: Accounts) -> RPS:
    logger.info("Preparing the RPS contract")
    
    # TODO: change following values to the correct ones
    rps_file = "ENTER_PATH_TO_RPS.sol"
    bytecode, abi = ["compile RPS.sol", "with provided function in part2/generation.py"]
    # TODO: change the following value to the correct one

    logger.info("Compiled. Now deploying the RPS contract")
    from_account = accounts[9]
    rps_contract = deploy_compiled_contract(
        w3, bytecode, abi, from_account, REVEAL_TIME)
    return RPS(w3, rps_contract)


@ pytest.fixture(scope="module")
def alice(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 1 ether from alice to the RPS contract")
    check_send_money(w3, accounts[0], rps.address, ONE_ETH)
    return accounts[0]


@ pytest.fixture(scope="module")
def bob(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 1 ether from bob to the RPS contract")
    check_send_money(w3, accounts[1], rps.address, ONE_ETH)
    return accounts[1]


@ pytest.fixture(scope="module")
def charlie(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 1 ether from charlie to the RPS contract")
    check_send_money(w3, accounts[2], rps.address, ONE_ETH)
    return accounts[2]

def get_commit(data: int, key: bytes) -> bytes:
    return bytes(Web3.solidity_keccak(['int256', 'bytes32'], [data, key]))

def test_two_game_flow(rps: RPS, alice: Any, bob: Any, charlie: Any) -> None:
    moves = [ROCK, PAPER, SCISSORS, ROCK]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]

    rps.make_move(1337, ONE_ETH//4, commits[0], from_account=alice)
    rps.make_move(17, ONE_ETH//4, commits[2], from_account=alice)
    rps.make_move(1337, ONE_ETH//4, commits[1], from_account=bob)
    rps.make_move(17, ONE_ETH//4, commits[3], from_account=charlie)

    rps.reveal_move(1337, moves[0], keys[0], from_account=alice)
    rps.reveal_move(17, moves[2], keys[2], from_account=alice)
    rps.reveal_move(1337, moves[1], keys[1], from_account=bob)

    rps.reveal_move(17, moves[3], keys[3], from_account=charlie)
    assert rps.get_game_state(1337) == NO_GAME
    assert rps.get_game_state(17) == NO_GAME

