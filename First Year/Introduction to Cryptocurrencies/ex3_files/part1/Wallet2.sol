// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// Food for thought: This wallet is not susceptible to reentrancy attacks. Can you explain why?
// (no need to submit an answer)

contract Wallet2 {
    mapping(address => uint) private userBalances;

    function deposit() external payable {
        //deposits eth into the account of the message sender.
        userBalances[msg.sender] += msg.value;
    }

    function sendTo(address payable destination, uint amount) external {
        //sends eth from the account of the message sender to the destination.
        require(amount >= userBalances[msg.sender]);
        (bool success, ) = destination.call{value: amount}("");
        require(success);
        userBalances[msg.sender] -= amount;
        // userBalances[msg.sender] = 0;
    }

    // This function allows you to fetch the balance of a user.
    function getBalance(address user) external view returns (uint) {
        return userBalances[user];
    } 
}
