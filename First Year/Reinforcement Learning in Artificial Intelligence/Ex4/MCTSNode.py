from ConnectFourTypes import *

class MCTSNode:
    def __init__(self, player_turn: Player, parent: 'MCTSNode', row: int, column: int):
        self.row = row
        self.column = column
        self.player_turn = player_turn
        self.parent: 'MCTSNode' = parent
        self.childs: list = self.init_childs(row, column)

        self.q: int = 0
        self.r: int = 0
        self.visits: int = 0
        self.cell_status: CellStatus = CellStatus.EMPTY

    def init_childs(self) -> list:
        self.childs = list()
        if self.column == 5:
            for i in range(7):
                self.childs.append(
                    MCTSNode(player_turn=Player.replace_player(self.player_turn), 
                                                parent=self,
                                                column=self.column + 1,
                                                row=i))
