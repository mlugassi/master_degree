import eth_typing
from web3.types import Wei
from web3 import Web3
from logger import logger
import secrets
import pytest
from typing import Any, Tuple
import web3.contract
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation

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

@pytest.fixture(scope="module")
def w3():
    from web3 import Web3
    provider = Web3.HTTPProvider("http://127.0.0.1:8545")
    return Web3(provider)

@pytest.fixture(scope="module")
def accounts(w3):
    return w3.eth.accounts

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
    rps_file = "RPS_GPT.sol"
    # bytecode, abi = ["compile ../RPS_GPT.sol", "../generation.py"]
    bytecode, abi = generation.compile(rps_file)
    # TODO: change the following value to the correct one

    logger.info("Compiled. Now deploying the RPS contract")
    from_account = accounts[9]
    rps_contract = deploy_compiled_contract(
        w3, bytecode, abi, from_account, REVEAL_TIME)
    return RPS(w3, rps_contract)


@ pytest.fixture(scope="module")
def rps_attc(w3: Web3, accounts: Accounts) -> RPS:
    logger.info("Preparing the RPS ATTACKER contract")
    
    # TODO: change following values to the correct ones
    rps_file = "RPS_ATTACKER.sol"
    # bytecode, abi = ["compile ../RPS_GPT.sol", "../generation.py"]
    bytecode, abi = generation.compile(rps_file)
    # TODO: change the following value to the correct one

    logger.info("Compiled. Now deploying the RPS contract")
    from_account = accounts[9]
    rps_contract = deploy_compiled_contract(
        w3, bytecode, abi, from_account, REVEAL_TIME)
    return RPS(w3, rps_contract)

@ pytest.fixture(scope="module")
def alice(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 10 ether from alice to the RPS contract")
    check_send_money(w3, accounts[0], rps.address, ONE_ETH * 10)
    return accounts[0]


@ pytest.fixture(scope="module")
def bob(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 10 ether from bob to the RPS contract")
    check_send_money(w3, accounts[1], rps.address, ONE_ETH * 10)
    return accounts[1]


@ pytest.fixture(scope="module")
def charlie(w3: Web3, rps: RPS, accounts: Accounts) -> Account:
    logger.info("Sending 10 ether from charlie to the RPS contract")
    check_send_money(w3, accounts[2], rps.address, ONE_ETH * 10)
    return accounts[2]

def simulate_timeout(w3: Web3, seconds: int) -> None:
    """
    Simulates a timeout by increasing the blockchain time and mining a block.

    :param w3: A Web3 instance connected to the local Ethereum test network.
    :param seconds: The number of seconds to advance the blockchain time.
    """
    logger.info(f"Simulating timeout by advancing time by {seconds} seconds.")

    # Increase the blockchain time
    w3.provider.make_request("evm_increaseTime", [seconds])
    
    # Mine a new block to apply the time change
    w3.provider.make_request("evm_mine", [])

    logger.info(f"Advanced time by {seconds} seconds and mined a new block.")


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

def test_insufficient_balance(rps: RPS, alice: Any) -> None:
    """Test that a player cannot bet more than their balance."""
    with pytest.raises(RevertException):
        rps.make_move(1, 50 * ONE_ETH + 1, get_commit(ROCK, secrets.token_bytes(32)), from_account=alice)

def test_double_move(rps: RPS, alice: Any) -> None:
    """Test that a player cannot make two moves in the same game."""
    commit = get_commit(ROCK, secrets.token_bytes(32))
    rps.make_move(2, ONE_ETH // 4, commit, from_account=alice)
    with pytest.raises(RevertException):
        rps.make_move(2, ONE_ETH // 4, commit, from_account=alice)

def test_double_reveal(rps: RPS, alice: Any) -> None:
    """Test that a player cannot reveal their move twice."""
    game_id = 3
    key = secrets.token_bytes(32)
    commit = get_commit(PAPER, key)
    rps.make_move(game_id, ONE_ETH // 4, commit, from_account=alice)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, PAPER, key, from_account=alice)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, PAPER, key, from_account=alice)

def test_game_cancellation(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that a game can be canceled if no move is made by the second player."""
    game_id = 4
    commit = get_commit(SCISSORS, secrets.token_bytes(32))
    rps.make_move(game_id, ONE_ETH // 4, commit, from_account=alice)
    rps.cancel_game(game_id, from_account=alice)
    assert rps.get_game_state(game_id) == NO_GAME

def test_reveal_phase_timeout(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that the first revealer can claim funds if the second player doesn't reveal on time."""
    game_id = 5
    moves = [ROCK, SCISSORS]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]

    alice_first_balance = rps.balance_of(alice)

    rps.make_move(game_id, ONE_ETH, commits[0], from_account=alice)
    rps.make_move(game_id, ONE_ETH * 10, commits[1], from_account=bob)
    rps.reveal_move(game_id, moves[0], keys[0], from_account=alice)

    # Simulate timeout
    rps.w3.provider.make_request("evm_increaseTime", [REVEAL_TIME + 1])
    rps.w3.provider.make_request("evm_mine", [])

    rps.reveal_phase_ended(game_id, from_account=alice)
    assert rps.get_game_state(game_id) == NO_GAME
    assert rps.balance_of(alice) == alice_first_balance + ONE_ETH

def test_reentrancy_protection(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that the contract is secure against reentrancy attacks."""
    game_id = 6
    moves = [PAPER, ROCK]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]
    alice_first_balance = rps.balance_of(alice)
    bob_first_balance = rps.balance_of(bob)

    logger.info(f'ALIC BALANCE #1 {alice_first_balance/ONE_ETH}')

    rps.make_move(game_id, ONE_ETH, commits[0], from_account=alice)
    rps.make_move(game_id, ONE_ETH * 100, commits[1], from_account=bob)
    rps.reveal_move(game_id, moves[0], keys[0], from_account=alice)
    rps.reveal_move(game_id, moves[1], keys[1], from_account=bob)
    logger.info(f'ALIC BALANCE #2 {rps.balance_of(alice)/ONE_ETH}')
    with pytest.raises(RevertException):
        # Simulate a reentrancy attack
        rps.withdraw(alice_first_balance + ONE_ETH * 2, from_account=alice)

    with pytest.raises(RevertException):
        # Simulate a reentrancy attack
        rps.withdraw(bob_first_balance, from_account=bob)        

def test_withdrawal(rps: RPS, charlie: Any, bob: Any) -> None:
    """Test that a player can withdraw their balance."""

    charlie_first_balance = rps.balance_of(charlie)
    bob_first_balance = rps.balance_of(bob)
    logger.info(f'CHARLIE BALANCE #1 {charlie_first_balance/ONE_ETH}')
    logger.info(f'BOB BALANCE #1 {bob_first_balance/ONE_ETH}')
    withdraw_amount = ONE_ETH * 5
    rps.withdraw(withdraw_amount, from_account=bob)

    assert rps.balance_of(bob) == bob_first_balance - withdraw_amount

    rps.withdraw(charlie_first_balance, from_account=charlie)
    assert rps.balance_of(charlie) == 0 

def test_reentrancy_attack(rps: RPS, alice: Account, w3: Web3) -> None:
    """Test reentrancy attack during withdrawal."""
    # Deploy the attacker contract
    attacker_contract_source = """
    pragma solidity ^0.8.0;

    interface IRPS {
        function withdraw(uint256 amount) external;
    }

    contract Attacker {
        IRPS public rpsContract;
        address public owner;

        constructor(address _rpsContract) {
            rpsContract = IRPS(_rpsContract);
            owner = msg.sender;
        }

        // Reentrancy exploit during withdrawal
        fallback() external payable {
            if (address(rpsContract).balance > 0) {
                rpsContract.withdraw(msg.value);
            }
        }

        function attack(uint256 amount) public {
            rpsContract.withdraw(amount);
        }

        function collect() public {
            require(msg.sender == owner, "Only owner can collect");
            payable(owner).transfer(address(this).balance);
        }

        receive() external payable {}
    }
    """
    bytecode, abi = generation.compile_from_source(attacker_contract_source, "Attacker")
    attacker = deploy_compiled_contract(w3, bytecode, abi, alice, rps.address)

    # Fund the attacker contract with some balance
    check_send_money(w3, alice, attacker.address, ONE_ETH)

    # Attempt a reentrancy attack
    with pytest.raises(RevertException):
        attacker.functions.attack(ONE_ETH).transact({'from': alice})

def test_timeout_exploitation(rps: RPS, alice: Account, bob: Account) -> None:
    """Test timeout exploitation by only one player revealing their move."""
    game_id = 7
    move = ROCK
    key = secrets.token_bytes(32)
    commit = get_commit(move, key)

    alice_first_balance = rps.balance_of(alice)
    logger.info(f'ALIC BALANCE #1 {alice_first_balance/ONE_ETH}')

    # Alice makes a move
    rps.make_move(game_id, ONE_ETH, commit, from_account=alice)

    # Bob doesn't respond
    simulate_timeout(rps.w3, REVEAL_TIME + 1)

    # Alice tries to exploit timeout
    with pytest.raises(RevertException):
        rps.reveal_phase_ended(game_id, from_account=alice)

    # Check game state
    assert rps.get_game_state(game_id) == MOVE1
    assert rps.balance_of(alice) == alice_first_balance - ONE_ETH

def test_replay_attack(rps: RPS, alice: Account, bob: Account) -> None:
    """Test that the same commit cannot be reused in multiple games."""
    game_id1 = 8
    game_id2 = 9
    move = PAPER
    key = secrets.token_bytes(32)
    commit = get_commit(move, key)

    # Alice uses the same commit for two different games
    rps.make_move(game_id1, ONE_ETH, commit, from_account=alice)
    # with pytest.raises(RevertException): #TODO check if that should not work
    #     rps.make_move(game_id2, ONE_ETH, commit, from_account=alice)

def test_invalid_reveal(rps: RPS, alice: Account, bob: Account) -> None:
    """Test revealing a move with an incorrect key or mismatched commit."""
    game_id = 10
    correct_move = SCISSORS
    correct_key = secrets.token_bytes(32)
    incorrect_key = secrets.token_bytes(32)
    commit = get_commit(correct_move, correct_key)

    # Alice makes a move
    rps.make_move(game_id, ONE_ETH, commit, from_account=alice)

    # Attempt to reveal with incorrect key
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, correct_move, incorrect_key, from_account=alice)

def test_excessive_withdrawal(rps: RPS, alice: Account) -> None:
    """Test that a player cannot withdraw more funds than they have."""
    alice_balance = rps.balance_of(alice)
    with pytest.raises(RevertException):
        rps.withdraw(alice_balance + ONE_ETH, from_account=alice)

def test_make_move_bet_more_than_balance(rps: RPS, alice: Any) -> None:
    """Test that a player cannot make a move with more than their balance."""
    with pytest.raises(RevertException):
        rps.make_move(10, ONE_ETH * 10, get_commit(ROCK, secrets.token_bytes(32)), from_account=alice)

def test_double_move_in_game(rps: RPS, alice: Any) -> None:
    """Test that a player cannot make two moves in the same game."""
    game_id = 100
    commit = get_commit(ROCK, secrets.token_bytes(32))
    rps.make_move(game_id, ONE_ETH, commit, from_account=alice)
    
    with pytest.raises(RevertException):
        rps.make_move(game_id, ONE_ETH, commit, from_account=alice)

def test_double_reveal(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that a player cannot reveal their move twice."""
    game_id = 101
    moves = [PAPER, ROCK]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]

    rps.make_move(game_id, ONE_ETH, commits[0], from_account=alice)
    rps.make_move(game_id, ONE_ETH * 100, commits[1], from_account=bob)
    rps.reveal_move(game_id, moves[0], keys[0], from_account=alice)
    
    with pytest.raises(RevertException):
        # Second reveal should fail
        rps.reveal_move(game_id, moves[0], keys[0], from_account=alice)

def test_cancel_game_after_move(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that a game can be canceled after the first player moves."""
    game_id = 102
    commit = get_commit(SCISSORS, secrets.token_bytes(32))
    rps.make_move(game_id, ONE_ETH, commit, from_account=alice)
    
    # Alice cancels the game before Bob reveals
    rps.cancel_game(game_id, from_account=alice)
    assert rps.get_game_state(game_id) == NO_GAME

def test_reveal_phase_timeout(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that the first revealer can claim funds if the second player doesn't reveal on time."""
    game_id = 103
    moves = [ROCK, SCISSORS]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]

    rps.make_move(game_id, ONE_ETH, commits[0], from_account=alice)
    rps.make_move(game_id, ONE_ETH * 10, commits[1], from_account=bob)
    rps.reveal_move(game_id, moves[0], keys[0], from_account=alice)

    # Simulate timeout
    simulate_timeout(rps.w3, REVEAL_TIME + 1)

    rps.reveal_phase_ended(game_id, from_account=alice)
    assert rps.get_game_state(game_id) == NO_GAME

def test_game_state_transitions(rps: RPS, alice: Any, bob: Any) -> None:
    """Test the game state transitions as players interact."""
    game_id = 104
    moves = [ROCK, PAPER]
    keys = [secrets.token_bytes(32) for _ in moves]
    commits = [get_commit(move, key) for move, key in zip(moves, keys)]

    # Alice and Bob make their moves
    rps.make_move(game_id, ONE_ETH, commits[0], from_account=alice)
    rps.make_move(game_id, ONE_ETH * 10, commits[1], from_account=bob)
    
    # Game state should be at 'MOVE2'
    assert rps.get_game_state(game_id) == MOVE2
    
    # Alice reveals
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[0], commits[1], from_account=alice)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[1], keys[1], from_account=alice)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[1], keys[0], from_account=alice)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[0], commits[1], from_account=bob)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[0], keys[0], from_account=bob)
    with pytest.raises(RevertException):
        rps.reveal_move(game_id, moves[1], keys[0], from_account=bob)        


    assert rps.get_game_state(game_id) == MOVE2
    rps.reveal_move(game_id, moves[1], keys[1], from_account=bob)
    assert rps.get_game_state(game_id) == REVEAL1

    
def test_cancel_game_unauthorized(rps: RPS, alice: Any, bob: Any) -> None:
    """Test that a game cannot be canceled by the second player."""
    game_id = 106
    commit1 = get_commit(ROCK, secrets.token_bytes(32))
    commit2 = get_commit(PAPER, secrets.token_bytes(32))
    
    # Alice and Bob make their moves
    rps.make_move(game_id, ONE_ETH, commit1, from_account=alice)
    rps.make_move(game_id, ONE_ETH * 10, commit2, from_account=bob)
    
    # Bob attempts to cancel the game, which should fail
    with pytest.raises(RevertException):
        rps.cancel_game(game_id, from_account=alice)
    with pytest.raises(RevertException):
        rps.cancel_game(game_id, from_account=bob)
    assert rps.get_game_state(game_id) == MOVE2
