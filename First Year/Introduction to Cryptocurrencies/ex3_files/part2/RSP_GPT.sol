// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

enum GameState {
    NO_GAME,
    MOVE1,
    MOVE2,
    REVEAL1,
    LATE
}

enum Move {
    NONE,
    ROCK,
    PAPER,
    SCISSORS
}

interface IRPS {
    function getGameState(uint gameID) external view returns (GameState);
    function makeMove(uint gameID, uint betAmount, bytes32 hiddenMove) external;
    function cancelGame(uint gameID) external;
    function revealMove(uint gameID, Move move, bytes32 key) external;
    function revealPhaseEnded(uint gameID) external;
    function balanceOf(address player) external view returns (uint);
    function withdraw(uint amount) external;
}

contract RPS is IRPS {
    struct Game {
        address player1;
        address player2;
        uint betAmount;
        bytes32 hiddenMove1;
        bytes32 hiddenMove2;
        Move revealedMove1;
        Move revealedMove2;
        uint revealBlock;
        GameState state;
    }

    uint public revealPeriodLength;
    mapping(uint => Game) public games;
    mapping(address => uint) public balances;

    constructor(uint _revealPeriodLength) {
        require(_revealPeriodLength >= 1, "Reveal period must be at least 1 block");
        revealPeriodLength = _revealPeriodLength;
    }

    function checkCommitment(
        bytes32 commitment,
        Move move,
        bytes32 key
    ) public pure returns (bool) {
        return keccak256(abi.encodePacked(uint(move), key)) == commitment;
    }

    function getGameState(uint gameID) external view override returns (GameState) {
        return games[gameID].state;
    }

    function makeMove(
        uint gameID,
        uint betAmount,
        bytes32 hiddenMove
    ) external override {
        Game storage game = games[gameID];

        require(game.state == GameState.NO_GAME || game.state == GameState.MOVE1, "Invalid game state");
        require(balances[msg.sender] >= betAmount, "Insufficient balance");

        if (game.state == GameState.NO_GAME) {
            game.player1 = msg.sender;
            game.hiddenMove1 = hiddenMove;
            game.betAmount = betAmount;
            game.state = GameState.MOVE1;
        } else {
            require(msg.sender != game.player1, "Player1 cannot play twice");
            require(betAmount == game.betAmount, "Bet amount mismatch");

            game.player2 = msg.sender;
            game.hiddenMove2 = hiddenMove;
            game.state = GameState.MOVE2;
        }

        balances[msg.sender] -= betAmount;
    }

    function cancelGame(uint gameID) external override {
        Game storage game = games[gameID];
        require(game.state == GameState.MOVE1, "Cannot cancel game at this stage");
        require(msg.sender == game.player1, "Only player1 can cancel the game");

        balances[game.player1] += game.betAmount;
        delete games[gameID];
    }

    function revealMove(uint gameID, Move move, bytes32 key) external override {
        Game storage game = games[gameID];

        require(game.state == GameState.MOVE2, "Game is not in reveal phase");
        require(msg.sender == game.player1 || msg.sender == game.player2, "Not a player in this game");

        if (msg.sender == game.player1) {
            require(game.revealedMove1 == Move.NONE, "Player1 has already revealed");
            require(checkCommitment(game.hiddenMove1, move, key), "Invalid commitment");
            game.revealedMove1 = move;
        } else {
            require(game.revealedMove2 == Move.NONE, "Player2 has already revealed");
            require(checkCommitment(game.hiddenMove2, move, key), "Invalid commitment");
            game.revealedMove2 = move;
        }

        if (game.revealedMove1 != Move.NONE && game.revealedMove2 != Move.NONE) {
            resolveGame(gameID);
        } else {
            game.state = GameState.REVEAL1;
            game.revealBlock = block.number;
        }
    }

    function revealPhaseEnded(uint gameID) external override {
        Game storage game = games[gameID];

        require(game.state == GameState.REVEAL1, "Game is not in reveal phase");
        require(block.number >= game.revealBlock + revealPeriodLength, "Reveal phase not ended");
        require(
            msg.sender == game.player1 || msg.sender == game.player2,
            "Only a player can end the reveal phase"
        );

        if (game.revealedMove1 != Move.NONE) {
            balances[game.player1] += 2 * game.betAmount;
        } else {
            balances[game.player2] += 2 * game.betAmount;
        }

        delete games[gameID];
    }

    function resolveGame(uint gameID) internal {
        Game storage game = games[gameID];

        if (
            (game.revealedMove1 == Move.ROCK && game.revealedMove2 == Move.SCISSORS) ||
            (game.revealedMove1 == Move.PAPER && game.revealedMove2 == Move.ROCK) ||
            (game.revealedMove1 == Move.SCISSORS && game.revealedMove2 == Move.PAPER)
        ) {
            balances[game.player1] += 2 * game.betAmount;
        } else if (
            (game.revealedMove2 == Move.ROCK && game.revealedMove1 == Move.SCISSORS) ||
            (game.revealedMove2 == Move.PAPER && game.revealedMove1 == Move.ROCK) ||
            (game.revealedMove2 == Move.SCISSORS && game.revealedMove1 == Move.PAPER)
        ) {
            balances[game.player2] += 2 * game.betAmount;
        } else {
            balances[game.player1] += game.betAmount;
            balances[game.player2] += game.betAmount;
        }

        delete games[gameID];
    }

    function balanceOf(address player) external view override returns (uint) {
        return balances[player];
    }

    function withdraw(uint amount) external override {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }

    receive() external payable {
        balances[msg.sender] += msg.value;
    }
}
