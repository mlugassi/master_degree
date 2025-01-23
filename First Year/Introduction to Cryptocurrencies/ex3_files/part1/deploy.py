from web3 import Web3
import solcx  # type: ignore
from typing import Any
from web3.types import Wei
import os

# run the line below to install the compiler ->  only once is needed.
# solcx.install_solc(version='latest')
# solcx.install_solc('0.8.19')

def compile(file_name: str) -> Any:
    solcx.set_solc_version('0.8.19')
    compiled_sol = solcx.compile_files([file_name], output_values=['abi', 'bin'])
    contract_id, contract_interface = compiled_sol.popitem()

    bytecode = contract_interface.get('bin', None)
    abi = contract_interface.get('abi', None)

    if not bytecode:
        raise ValueError(f"No bytecode generated for {file_name}. Check for issues in the Solidity file.")

    return bytecode, abi

# Change the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# bytecode, abi = compile("greeter.sol")


# Connect to the blockchain: (Hardhat node should be running at this port)
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# deploy the contract
# Greeter = w3.eth.contract(abi=abi, bytecode=bytecode)

# bytecode, abi = compile("./Wallet2.sol")
bytecode, abi = compile("./VulnerableWallet.sol")
assert bytecode, "Bytecode is empty0"

Vul = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = Vul.constructor().transact(
    {'from': w3.eth.accounts[0]})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

vul = w3.eth.contract(address=tx_receipt["contractAddress"], abi=abi)

Abytecode, Aabi = compile("./WalletAttack.sol")
assert Abytecode, "Bytecode is empty"
Attacker = w3.eth.contract(abi=Aabi, bytecode=Abytecode)
tx_hash1 = Attacker.constructor().transact(
    {'from': w3.eth.accounts[1]})
tx_receipt1 = w3.eth.wait_for_transaction_receipt(tx_hash1)
attacker = w3.eth.contract(address=tx_receipt1["contractAddress"], abi=Aabi)

# print("the wallet 10 balance is:", float(w3.eth.get_balance(w3.eth.accounts[10])/(10**18)))

for _ in range(2):
    _ = vul.functions.deposit().transact( {'from': w3.eth.accounts[2], 'value': w3.to_wei(1, 'ether')})
print("the wallet 2 balance before is:", float(w3.eth.get_balance(w3.eth.accounts[2])/(10**18)))
print("the wallet 1 balance before is:", float(w3.eth.get_balance(w3.eth.accounts[1])/(10**18)))
print("the attacker balance before is:", float(w3.eth.get_balance(attacker.address)/(10**18)))
print("the vul balance before is:", float(w3.eth.get_balance(vul.address)/(10**18)))
# tx_hash = vul.functions.sendTo(w3.eth.accounts[1], w3.to_wei(1, 'ether')).transact( {'from': w3.eth.accounts[2]})
tx_hash2 = attacker.functions.exploit(vul.address).transact( {'from': w3.eth.accounts[1], 'value': w3.to_wei(1, 'ether')})
# tx_hash2 = attacker.functions.exploit(vul.address).transact( {'from': w3.eth.accounts[1], 'value': w3.to_wei(1, 'ether')})
print("the wallet 1 balance after is:", float(w3.eth.get_balance(w3.eth.accounts[1])/(10**18)))
print("the wallet 2 balance after is:", float(w3.eth.get_balance(w3.eth.accounts[2])/(10**18)))
# print(f"userBalances account 1:", float(vul.functions.getBalance(w3.eth.accounts[1]).call({'from': w3.eth.accounts[1]})/(10**18)))
# print(f"userBalances account 2:", float(vul.functions.getBalance(w3.eth.accounts[2]).call({'from': w3.eth.accounts[2]})/(10**18)))
# print(f"userBalances attacker:", float(vul.functions.getBalance(attacker.address).call({'from': attacker.address})/(10**18)))
print("the attacker balance after is:", float(w3.eth.get_balance(attacker.address)/(10**18)))
print("the vul balance after is:", float(w3.eth.get_balance(vul.address)/(10**18)))



