//SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./ChannelInterface.sol";

enum ChannelState {
    OPEN,
    APPEAL_PERIOD,
    CLOSE
} 



contract Channel is ChannelI {
    // This contract will be deployed every time we establish a new payment channel between two participant.
    // The creator of the channel also injects funds that can be sent (and later possibly sent back) in this channel
    address payable public first_owner;
    address payable public other_owner;
    uint public appeal_period_len;
    uint public balance1;
    uint public balance2;
    uint public serial_num;
    uint256 public close_time;
    bool public got_close_req;

    function getFirstOwner() external view returns (address) {
        return first_owner;
    }

    function getOtherOwner() external view returns (address) {
        return other_owner;
    }

    function getSerialNum() external view returns(uint) {
        return serial_num;
    }

    function getAppealPeriodLen() external view returns(uint) {
        return appeal_period_len;
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
        first_owner = payable(msg.sender);
        other_owner = _otherOwner;
        appeal_period_len = _appealPeriodLen;
        balance1 = msg.value;
        balance2 = 0;
        close_time = 0;
        got_close_req = false;
    }

    // IMPLEMENT ADDITIONAL FUNCTIONS HERE
    // See function definitions in the interface ChannelI.
    // Make sure to implement all of the functions from the interface ChannelI.
    // Define your own state variables, and any additional functions you may need in addition to that...


    function getChannelState() external view returns (ChannelState) {
        if (got_close_req == false) {
            return ChannelState.OPEN;
        } else if (block.timestamp <= appeal_period_len + close_time) {
            return ChannelState.APPEAL_PERIOD;
        } else {
            return ChannelState.CLOSE;
        }   
    }

    //Closes the channel based on a message by one party.
    //If the serial number is 0, then the provided balance and signatures are ignored, and the channel is closed according to the initial split, 
    //giving all the money to party 1.
    //Closing the channel starts the appeal period.
    // If any of the parameters are bad (signature,balance) the transaction reverts.
    // Additionally, the transactions would revert if the party closing the channel isn't one of the two participants.
    // _balance1 is the balance that belongs to the user that opened the channel. _balance2 is for the other user.
    function oneSidedClose(
        uint _balance1,
        uint _balance2,
        uint serialNum,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        require(msg.sender == first_owner || msg.sender == other_owner, "Invalid address");
        require(this.getChannelState() == ChannelState.OPEN, "Channel is already closed");
        require(serialNum >= 0, "Invalid serial number");

        if (serialNum == 0) {
            balance1 = balance1 + balance2;
            balance2 = 0;
            serial_num = 0;
        } else {
            if (msg.sender == first_owner) {
                require(_verifySig(address(this), _balance1, _balance2, serialNum, v, r, s, other_owner), "Invalid signature");
            } else {
                require(_verifySig(address(this), _balance1, _balance2, serialNum, v, r, s, first_owner), "Invalid signature");
            }
            require((_balance1 + _balance2) == (balance1 + balance2), "Invalid balance");
            require((_balance1 >= 0) && (_balance2 >= 0), "Invalid balance");

            balance1 = _balance1;
            balance2 = _balance2;
            serial_num = serialNum;
        }

        close_time = block.timestamp;
        got_close_req = true;
    }

    // appeals a one_sided_close. should show a signed message with a higher serial number.
    // _balance1 belongs to the creator of the contract. _balance2 is the money going to the other user.
    // this function reverts upon any problem:
    // It can only be called during the appeal period.
    // only one of the parties participating in the channel can appeal.
    // the serial number, balance, and signature must all be provided correctly.
    function appealClosure(
        uint _balance1,
        uint _balance2,
        uint serialNum,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        require(msg.sender == first_owner || msg.sender == other_owner, "Invalid address");
        require(this.getChannelState() == ChannelState.APPEAL_PERIOD, "Channel already closed or still open");
        require(serialNum > serial_num, "Invalid serial number");

        if (msg.sender == first_owner) {
            require(_verifySig(address(this), _balance1, _balance2, serialNum, v, r, s, other_owner), "Invalid signature");
        } else {
            require(_verifySig(address(this), _balance1, _balance2, serialNum, v, r, s, first_owner), "Invalid signature");
        }
        require((_balance1 + _balance2) == (balance1 + balance2), "Invalid balance");
        require((_balance1 >= 0) && (_balance2 >= 0), "Invalid balance");        

        balance1 = _balance1;
        balance2 = _balance2;
        serial_num = serialNum;

        close_time = block.timestamp;
        got_close_req = true;    
    }

    // Sends all of the money belonging to msg.sender to the destination address provided.
    // this should only be possible if the channel is closed, and appeals are over.
    // This transaction should revert upon any error.
    function withdrawFunds(address payable destAddress) external {

        require(msg.sender == first_owner || msg.sender == other_owner, "Invalid address");
        require(this.getChannelState() == ChannelState.CLOSE, "Channel isn't closed yet");

        if (msg.sender == first_owner) {
            require(balance1 > 0, "No funds to withdraw");
            uint balance = balance1;
            balance1 = 0;
            (bool success, ) = destAddress.call{value: balance}("");
            require(success, "Transfer failed");
        } else if (msg.sender == other_owner) {
            require(balance2 > 0, "No funds to withdraw");
            uint balance = balance2;
            balance2 = 0;
            (bool success, ) = destAddress.call{value: balance}("");
            require(success, "Transfer failed");
        }
    }

    // returns the balance of the caller (the funds that this person can withdraw) if he is one of the channel participants.
    // This function should revert if the channel is still open, or if the appeal period has not yet ended.
    function getBalance() external view returns (uint) {
        require(msg.sender == first_owner || msg.sender == other_owner, "Invalid address");
        require(this.getChannelState() == ChannelState.CLOSE, "Channel isn't closed yet");
        if (msg.sender == first_owner) {
            return balance1;
        } else {
            return balance2;
        }
    }
}
