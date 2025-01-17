from web3 import Web3
import solcx  # type: ignore
from typing import Any
from web3.types import Wei
import os
# run the line below to install the compiler ->  only once is needed.
# solcx.install_solc(version='0.8.19')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def compile(file_name: str) -> Any:
    # set the version
    solcx.set_solc_version('0.8.19')

    # compile
    compiled_sol = solcx.compile_files(
        [file_name], output_values=['abi', 'bin'])

    # retrieve the contract interface
    contract_id, contract_interface = compiled_sol.popitem()
    return contract_interface['bin'], contract_interface['abi']

# contract = "RSP"
contract = "RPS_GPT"
bytecode, abi = compile(f"{contract}.sol")
with open(f"{contract}.abi", "w") as f:
    for line in abi:
        f.write(str(line))
        f.write("\n")

with open(f"{contract}.bin", "w") as f:
    f.write(bytecode)


import solcx

def compile_from_source(source_code: str, contract_name: str):
    """
    Compile a Solidity contract from its source code.

    :param source_code: The Solidity source code as a string.
    :param contract_name: The name of the contract to compile.
    :return: A tuple of (bytecode, abi) for the specified contract.
    """
    solcx.set_solc_version('0.8.19')

    # Compile the source code
    compiled_contracts = solcx.compile_source(
        source_code,
        output_values=["abi", "bin"],
    )

    # Extract the bytecode and ABI for the specified contract
    contract_data = compiled_contracts[f"<stdin>:{contract_name}"]
    bytecode = contract_data["bin"]
    abi = contract_data["abi"]

    return bytecode, abi
