from web3 import Web3
import solcx  # type: ignore
from typing import Any
from web3.types import Wei
import os

# run the line below to install the compiler ->  only once is needed.
# solcx.install_solc(version='latest')
# solcx.install_solc('0.8.19')

def compile(file_name: str) -> Any:
    # set the version
    solcx.set_solc_version('0.8.19')

    # compile
    compiled_sol = solcx.compile_files(
        [file_name], output_values=['abi', 'bin'])

    # retrieve the contract interface
    contract_id, contract_interface = compiled_sol.popitem()
    return contract_interface['bin'], contract_interface['abi']

script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script directory
os.chdir(script_dir)
# bytecode, abi = compile("greeter.sol")


# Connect to the blockchain: (Hardhat node should be running at this port)
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# deploy the contract
# Greeter = w3.eth.contract(abi=abi, bytecode=bytecode)

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

for _ in range(5):
    _ = vul.functions.deposit().transact( {'from': w3.eth.accounts[2], 'value': w3.to_wei(1, 'ether')})
print("the wallet 2 balance before is:", float(w3.eth.get_balance(w3.eth.accounts[2])/(10**18)))

print("the wallet 1 balance before is:", float(w3.eth.get_balance(w3.eth.accounts[1])/(10**18)))
tx_hash2 = attacker.functions.exploit(vul.address).transact( {'from': w3.eth.accounts[1], 'value': w3.to_wei(1, 'ether')})
print("the wallet 1 balance after is:", float(w3.eth.get_balance(w3.eth.accounts[1])/(10**18)))
print("the wallet 2 balance after is:", float(w3.eth.get_balance(w3.eth.accounts[2])/(10**18)))




# tx_hash = vul.functions.sendTo(w3.eth.accounts[5]).transact( {'from': w3.eth.accounts[10]})

# print("the VUL balance is:", float(w3.eth.get_balance(vul.address))/(10**18))
# print("the wallet 5 balance is:", float(w3.eth.get_balance(w3.eth.accounts[5]))/(10**18))
# print("the wallet 10 balance is:", float(w3.eth.get_balance(w3.eth.accounts[10])/(10**18)))

# tx_hash = w3.eth.send_transaction({
#     'to': w3.eth.accounts[2],
#     'from': w3.eth.accounts[3],  # type: ignore
#     'value': Wei(10**16)
# })
# tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
# print("the wallet 2 balance is:", float(w3.eth.get_balance(w3.eth.accounts[2,l;])/(10**18)))
# print("the wallet 3 balance is:", float(w3.eth.get_balance(w3.eth.accounts[3])/(10**18)))

# Submit the transaction that deploys the contract. It is deployed by accounts[0] which is the first of the 10 pre-made accounts created by hardhat.
# tx_hash = Greeter.constructor("Hello!").transact(
#     {'from': w3.eth.accounts[1]})

# # Wait for the transaction to be mined, and get the transaction receipt
# tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# # get a contract instance
# print(tx_receipt)
# greeter = w3.eth.contract(address=tx_receipt["contractAddress"], abi=abi)

# # here we call a view function (that does not require a transaction to the blockchain). This is done via '.call()'
# print(greeter.functions.greet().call())

# # here we call a function that changes the state and does require a blockchain transaction. This is done via '.transact()'
# tx_hash = greeter.functions.setGreeting(
#     'Nihao').transact({"from": w3.eth.accounts[5]})  # type: ignore

# # wait for a transaction to be mined.
# tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# # check the greeting again.
# print(greeter.functions.greet().call())

# try:
#     tx_hash = w3.eth.send_transaction({
#         'to': greeter.address,
#         'from': w3.eth.accounts[1],  # type: ignore
#         'value': Wei(10**16)
#     })
#     tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
# except:
#     pass
# print("the contract's balance is:", w3.eth.get_balance(greeter.address))

# # now we withdraw:
# tx_hash = greeter.functions.withdraw().transact(
#     {"from": w3.eth.accounts[2]})  # type: ignore
# tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# # print("the contract's balance is:", w3.eth.get_balance(greeter.address))
# # print("account 2 now has:", w3.eth.get_balance(
# #     w3.eth.accounts[2]))  # type: ignore
