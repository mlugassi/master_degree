// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// import "../hardhat/console.sol";

enum GameState {
    NO_GAME, //signifies that there is no game with this id (or there was and it is over)
    MOVE1, //signifies that a single move was entered
    MOVE2, //a second move was enetered
    REVEAL1, //one of the moves was revealed, and the reveal phase just started
    LATE // one of the moves was revealed, and enough blocks have been mined since so that the other player is considered late.
} // These correspond to values 0,1,2,3,4

enum Move {
    NONE,
    ROCK,
    PAPER,
    SCISSORS
} //These correspond to values 0,1,2,3

interface IRPS {
    // WARNING: Do not change this interface!!! these API functions are used to test your code.
    function getGameState(uint gameID) external view returns (GameState);
    function makeMove(uint gameID, uint betAmount, bytes32 hiddenMove) external;
    function cancelGame(uint gameID) external;
    function revealMove(uint gameID, Move move, bytes32 key) external;
    function revealPhaseEnded(uint gameID) external;
    function balanceOf(address player) external view returns (uint);
    function withdraw(uint amount) external;
}

contract RPS is IRPS {
    // This contract lets players play rock-paper-scissors.
    // its constructor receives a uint k which is the number of blocks mined before a reveal phase is over.

    // players can send the contract money to fund their bets, see their balance and withdraw it, as long as the amount is not in an active game.

    // the game mechanics: The players choose a gameID (some uint) that is not being currently used. They then each call make_move() making a bet and committing to a move.
    // in the next phase each of them reveals their committment, and once the second commit is done, the game is over. The winner gets the amount of money they agreed on.

    //TODO: add state variables and additional functions as needed.

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

    uint public immutable revealPeriodLength;
    mapping(uint => Game) public games;
    mapping(address => uint) public balances;

    constructor(uint _revealPeriodLength) {
        // Constructs a new contract that allows users to play multiple rock-paper-scissors games.
        // If one of the players does not reveal the move committed to, then the _revealPeriodLength
        // is the number of blocks that a player needs to wait from the moment of revealing her move until
        // she can calim that the other player loses (for not revealing).
        // The _revealPeriodLength must be at least 1 block.
        require(_revealPeriodLength >= 1, "Reveal period must be at least 1 block");
        revealPeriodLength = _revealPeriodLength;
    }

    function checkCommitment(
        bytes32 commitment,
        Move move,
        bytes32 key
    ) public pure returns (bool) {
        // A utility function that can be used to check commitments. See also commit.py.
        // python code to generate the commitment is:
        // commitment = HexBytes(Web3.solidityKeccak(['int256', 'bytes32'], [move, key]))
        return keccak256(abi.encodePacked(uint(move), key)) == commitment;
    }

    function getGameState(uint gameID) external view returns (GameState) {
        // Returns the state of the game at the current address as a GameState (see enum definition)
        // if(games[gameID].exists)  //TODO do not forget to check of we can know if key is exsits in a mapping
        return games[gameID].state;        
    }

    function makeMove(
        uint gameID,
        uint betAmount,
        bytes32 hiddenMove
     ) external {
        // The first call to this function starts the game. The second call finishes the commit phase.
        // The amount is the amount of money (in wei) that a user is willing to bet.
        // The amount provided in the call by the second player is ignored, but the user must have an amount matching that of the game to bet.
        // amounts that are wagered are locked for the duration of the game.
        // A player should not be allowed to enter a commitment twice.
        // If two moves have already been entered, then this call reverts.
        Game storage game = games[gameID];
        require(game.state == GameState.NO_GAME || game.state == GameState.MOVE1, "Invalid game state");

        if(game.state == GameState.MOVE1) {
            require(balances[msg.sender] >= game.betAmount, "Insufficient balance");
        } else {
            require(balances[msg.sender] >= betAmount, "Insufficient balance");
        }

        if (game.state == GameState.NO_GAME) {
            game.player1 = msg.sender;
            game.hiddenMove1 = hiddenMove;
            game.betAmount = betAmount;
            game.state = GameState.MOVE1;
        } else {
            require(msg.sender != game.player1, "Player1 cannot play twice");
            game.player2 = msg.sender;
            game.hiddenMove2 = hiddenMove;
            game.state = GameState.MOVE2;
        }
        balances[msg.sender] -= game.betAmount;
    }

    function getGameStateString(GameState _state) internal pure returns (string memory) {
        if (_state == GameState.NO_GAME) return "NO_GAME";
        if (_state == GameState.MOVE1) return "MOVE1";
        if (_state == GameState.MOVE2) return "MOVE2";
        if (_state == GameState.REVEAL1) return "REVEAL1";
        if (_state == GameState.LATE) return "LATE";
        return "UNKNOWN";
    }

    function cancelGame(uint gameID) external {
        // This function allows a player to cancel the game, but only if the other player did not yet commit to his move.
        // a canceled game returns the funds to the player. Only the player that made the first move can call this function, and it will run only if
        // no other commitment for a move was entered. This function reverts in any other case.
        Game storage game = games[gameID];
        require(msg.sender == game.player1 && game.state == GameState.MOVE1, "Cancel can be done only by player1 and in MOVE1 state");
        game.state = GameState.NO_GAME;
        balances[game.player1] += game.betAmount;
        delete games[gameID];
    }

    function revealMove(uint gameID, Move move, bytes32 key) external {
        // reveals the move of a player (which is checked against his commitment using the key)
        // The first call can be made only after two moves are entered.
        // it will begin the reveal phase that ends in k blocks.
        // the second successful call ends the game and awards the money to the winner.
        // each player is allowed to reveal only once and only the two players that entered moves may reveal.
        // this function reverts on any other case and in any case of failure to properly reveal.
        Game storage game = games[gameID];

        require(game.state == GameState.MOVE2 || game.state == GameState.REVEAL1, "Game is not in reveal phase");
        require(msg.sender == game.player1 || msg.sender == game.player2, "Not a player in this game");

        if (msg.sender == game.player1) {
            require(game.revealedMove1 == Move.NONE, "Player1 has already revealed");
            require(checkCommitment(game.hiddenMove1, move, key), "Invalid commitment");
            require(move == Move.PAPER || move == Move.ROCK || move == Move.SCISSORS, "The move isn't allowed");
            game.revealedMove1 = move;
        } else {
            require(game.revealedMove2 == Move.NONE, "Player2 has already revealed");
            require(checkCommitment(game.hiddenMove2, move, key), "Invalid commitment");
            require(move == Move.PAPER || move == Move.ROCK || move == Move.SCISSORS, "The move isn't allowed");
            game.revealedMove2 = move;
        }

        if (game.revealedMove1 != Move.NONE && game.revealedMove2 != Move.NONE) {
            resolveGame(gameID);
        } else {
            game.state = GameState.REVEAL1;
            game.revealBlock = block.timestamp;
        }
    }

    function revealPhaseEnded(uint gameID) external {
        // If no second reveal is made, and the reveal period ends, the player that did reveal can claim all funds wagered in this game.
        // The game then ends, and the game id is released (and can be reused in another game).
        // this function can only be called by the first revealer. If the reveal phase is not over, this function reverts.
        Game storage game = games[gameID];

        require(game.state == GameState.REVEAL1, "Game is not in reveal phase");
        require(block.timestamp >= game.revealBlock + revealPeriodLength, "Reveal phase not ended");
        require(msg.sender == game.player1 || msg.sender == game.player2, "Only a player can end the reveal phase");
        require((game.revealedMove1 != Move.NONE && game.revealedMove2 == Move.NONE) || 
                 (game.revealedMove1 == Move.NONE && game.revealedMove2 != Move.NONE), "Only one reveal must be done");
        require((msg.sender == game.player1 && game.revealedMove1 != Move.NONE) || 
        (msg.sender == game.player2 && game.revealedMove2 != Move.NONE), "Reveal can be done only by first revealer");
        game.state = GameState.LATE;
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
    
    ////////// Handling money ////////////////////
    function balanceOf(address player) external view returns (uint) {
        // returns the balance of the given player. Funds that are wagered in games that did not complete yet are not counted as part of the balance.
        // make sure the access level of this function is "view" as it does not change the state of the contract.
        return balances[player];
    }

    function withdraw(uint amount) external {
        // Withdraws amount from the account of the sender
        // (available funds are those that were deposited or won but not currently staked in a game).
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }

    receive() external payable {
        require(msg.value >= 0, "Can't recieve negative amount");
        balances[msg.sender] += msg.value;
    }
}
