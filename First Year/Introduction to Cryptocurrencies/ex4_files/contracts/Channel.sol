//SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./ChannelInterface.sol";

contract Channel is ChannelI {
    // This contract will be deployed every time we establish a new payment channel between two participant.
    // The creator of the channel also injects funds that can be sent (and later possibly sent back) in this channel
    address payable public first_owner;
    address payable public other_owner;
    uint public appeal_period_len;
    uint public balance1;
    uint public balance2;
    uint public serial_num;


    function getFirstOwner() external view returns (address) {
        return this.first_owner;
    }

    function getOtherOwner() external view returns (address) {
        return this.other_owner;
    }

    function  getAppealPeriodLen() external view returns (uint) {
        return this.appeal_period_len;
    }

    function getMyBalance() external view returns (uint) {
        if (msg.sender == this.first_owner) {
            return this.balance1;
        } else if (msg.sender == this.other_owner) {
            return this.balance2;
        } else {
            return 0;
        }
    }
    
    function getBalance1() external view returns (uint) {
        return this.balance1;
    }

    function getBalance2() external view returns (uint) {
        return this.balance2;
    }

    function getSerialNum() external view returns (uint) {
        return this.serial_num;
    }

    function _verifySig(
        // Do not change this function!
        address contract_address,
        uint _balance1,
        uint _balance2,
        uint serialNum, //<--- the message
        uint8 v,
        bytes32 r,
        bytes32 s, // <---- The signature
        address signerPubKey
    ) public pure returns (bool) {
        // v,r,s together make up the signature.
        // signerPubKey is the public key of the signer
        // contract_address, _balance1, _balance2, and serialNum constitute the message to be signed.
        // returns True if the sig checks out. False otherwise.

        // the message is made shorter:
        bytes32 hashMessage = keccak256(
            abi.encodePacked(contract_address, _balance1, _balance2, serialNum)
        );

        //message signatures are prefixed in ethereum.
        bytes32 messageDigest = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", hashMessage)
        );
        //If the signature is valid, ecrecover ought to return the signer's pubkey:
        return ecrecover(messageDigest, v, r, s) == signerPubKey;
    }

    constructor(address payable _otherOwner, uint _appealPeriodLen) payable {
        // Do not change the signature of this constructor! Implement the logic inside.
        this.first_owner = msg.sender;
        this.other_owner = _otherOwner;
        this.appeal_period_len = _appealPeriodLen;
        this.balance1 = msg.value;
        this.balance2 = 0;
    }

    // IMPLEMENT ADDITIONAL FUNCTIONS HERE
    // See function definitions in the interface ChannelI.
    // Make sure to implement all of the functions from the interface ChannelI.
    // Define your own state variables, and any additional functions you may need in addition to that...

    function oneSidedClose(
        uint _balance1,
        uint _balance2,
        uint serialNum,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        // TODO
    }

    function appealClosure(
        uint _balance1,
        uint _balance2,
        uint serialNum,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        // TODO
    }

    function withdrawFunds(address payable destAddress) external {
        // TODO
        require(msg.sender == destAddress, "Only the owner can withdraw funds");
        require(destAddress == this.first_owner || destAddress == this.other_owner, "Invalid address");

        if (destAddress == this.first_owner) {
            require(this.balance1 > 0, "No funds to withdraw");
            uint balance = this.balance1;
            this.balance1 = 0;
            (bool success, ) = destAddress.call{value: amount}("");
            require(success, "Transfer failed");
        } else if (destAddress == this.other_owner) {
            require(this.balance2 > 0, "No funds to withdraw");
            uint balance = this.balance2;
            this.balance2 = 0;
            (bool success, ) = destAddress.call{value: amount}("");
            require(success, "Transfer failed");
        }
    }

    function getBalance() external view returns (uint) {
        return this.balance1 + this.balance2
    }
}
