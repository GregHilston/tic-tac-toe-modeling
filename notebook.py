#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import pandas as pd
from enum import Enum
from interface import implements, Interface
import random
import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[10]:


df = pd.read_csv("train.csv")


# In[11]:


df.info()


# In[12]:


df.head()


# In[54]:


class Space(Enum):
    """Represents a single space by a player or an empty spot

    """
    def to_float(self):
        """So Scikit learn can interpret
        
        """
        if self.value == 1:
            return 0.0
        elif self.value == 2:
            return 1.0
        elif self.value == 3:
            return 2.0
    
    def __str__(self):
        if self.value == 1:
            return '_'
        elif self.value == 2:
            return 'x'
        elif self.value == 3:
            return 'o'
        else:
            return '?'

    EMPTY = 1
    X = 2
    O = 3


# In[58]:


class Move():
    """Represents a single attempted move by a player
    
    """
    def __init__(self, space: Space, index: int):
            self.space = space
            self.index = index
            
    def __repr__(self):
        return f"Player {self.space} attempted to move at index {self.index}"


# In[81]:


class PlayerStrategy(Interface):
    """Defines a purely random strategy
    
    """
    def __init__(self, space: Space):
        pass

    def query_move(self, board: "Board", first_query_failed: bool = False) -> Move:
        pass


# In[82]:


class RandomPlayerStrategy(implements(PlayerStrategy)):
    """Defines a purely random strategy
    
    """
    def __init__(self, space: Space):
        self.space = space

    def query_move(self, board: "Board", first_query_failed: bool = False) -> Move:
        return Move(self.space, random.randint(0, 8)) # hardcoded 9 here, its fine


# In[83]:


class HumanPlayerStrategy(implements(PlayerStrategy)):
    """Defines a purely random strategy
    
    """
    def __init__(self, space: Space):
        self.space = space

    def query_move(self, board: "Board", first_query_failed: bool = False) -> Move:
        print("Your move, current board:")
        print(str(type(board)))
        print(board)
        print(f"Place a {self.space} at index[0-8]:")
        valid_input = False
        
        while not valid_input:
            try:
                user_input = input()
                index = int(user_input)
                if index < 0 or index > 8:
                    print(f"You entered an invalid index {user_input}, please enter one between [0-8]")
                else:
                    valid_input = True
            except:
                print(f"You entered an invalid index {user_input}, please enter one between [0-8]")
              
        print(f"you entered {index}")
              
        return Move(self.space, index)


# In[106]:


class DecisionTreeClassifierPlayerStrategy(implements(PlayerStrategy)):
    def __init__(self, space: Space):
        self.space = space
        self.decision_tree_classifier = DecisionTreeClassifier(random_state=42)
        df = pd.read_csv("train_2.csv")
        x = df.drop("move", axis=1)# df[["index_0", "index_1", "index_2", "index_3", "index_4", "index_5", "index_6", "index_7", "index_8", "whose_turn"]] # Features
        y = df["move"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
        self.random_player_strategy = RandomPlayerStrategy(self.space)

        self.decision_tree_classifier = self.decision_tree_classifier.fit(x_train, y_train)

    def query_move(self, board: "Board", first_query_failed: bool = False) -> Move:
        print(f"first_query_failed {first_query_failed}")
        # if we can't do what we want, we'll randomize
        if first_query_failed:
            print("DecisionTreeClassifierPlayerStrategy failed, asking Random")
            return self.random_player_strategy.query_move(board, first_query_failed)
        
        print("DecisionTreeClassifierPlayerStrategy attempt")

        index = self.decision_tree_classifier.predict([[
            board.board[0].to_float(), board.board[1].to_float(), board.board[2].to_float(),
            board.board[3].to_float(), board.board[4].to_float(), board.board[5].to_float(),
            board.board[6].to_float(), board.board[7].to_float(), board.board[8].to_float(),
            self.space.to_float()
        ]])

        return Move(self.space, index[0])


# In[107]:


class MoveWritter:
    def __init__(self, file_name: str):
        self.file_name = file_name
        
        # potentially write header
        if not os.path.isfile(self.file_name):
            print("File doesn't exist, writing header!")
            with open(self.file_name, 'w') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "index_0", 
                    "index_1", 
                    "index_2", 
                    "index_3", 
                    "index_4", 
                    "index_5", 
                    "index_6", 
                    "index_7",
                    "index_8",
                    "whose_turn",
                    "move"
                ])
        else:
            print("File does exist, NOT writing header!")


    def write_move(self, board: "Board", is_player_x_turn: bool, move: int):
        with open(self.file_name, 'a', newline='') as file:
            if is_player_x_turn:
                player = 'x'
            else:
                player = 'o'
            
            writer = csv.writer(file)
            writer.writerow([
                str(board[0]),
                str(board[1]),
                str(board[2]),
                str(board[3]),
                str(board[4]),
                str(board[5]),
                str(board[6]),
                str(board[7]),
                str(board[8]),
                player,
                move
            ])


# In[114]:


class Board:
    """Represents as tic tac toe board
    0 1 2
    3 4 5
    6 7 8
    
    """
    
    class MoveAlreadyTakenException(Exception):
        """Represents an exception when an already made move was attempted
        again.
        
        """
        pass

    def __init__(self, player_x: PlayerStrategy, player_o: PlayerStrategy, move_writer: MoveWritter = None):
        self.player_x = player_x
        self.player_o = player_o
        self.board = [
            Space.EMPTY, Space.EMPTY, Space.EMPTY,
            Space.EMPTY, Space.EMPTY, Space.EMPTY,
            Space.EMPTY, Space.EMPTY, Space.EMPTY
        ]
        self.move_writer = move_writer
    
    def _attempt_move(self, move: Move):
        """Attemts to perform a single move
        
        """
        
        if self.board[move.index] != Space.EMPTY:
            raise Board.MoveAlreadyTakenException()
            
        self.board[move.index] = move.space
            
    def _get_winner(self):
        """Determining who has won. 
        Bad code ahead!
        
        """
        def _set_contain_winner(set_to_check):
            """Checks if this set (horizontal, vertical or diagnol) is a winner
            
            """
            if Space.EMPTY not in set_to_check and len(set_to_check) == 1:
                return True
            return False
        
        horizontals = []
        verticals = []
        diagnols = []
        
        horizontals.append(set([self.board[0], self.board[1], self.board[2]]))            
        horizontals.append(set([self.board[3], self.board[4], self.board[5]]))            
        horizontals.append(set([self.board[6], self.board[7], self.board[8]]))         
        
        verticals.append(set([self.board[0], self.board[3], self.board[6]]))            
        verticals.append(set([self.board[1], self.board[5], self.board[7]]))            
        verticals.append(set([self.board[2], self.board[5], self.board[8]]))   
        
        diagnols.append(set([self.board[0], self.board[4], self.board[8]]))            
        diagnols.append(set([self.board[2], self.board[4], self.board[6]]))  
        
        for horizontal in horizontals:
            if(_set_contain_winner(horizontal)):
                return horizontal.pop() # just to get to element, set is useless now anyways
            
        for vertical in verticals:
            if(_set_contain_winner(vertical)):
                return vertical.pop() # just to get to element, set is useless now anyways
            
        for diagnol in diagnols:
            if(_set_contain_winner(diagnol)):
                return diagnol.pop() # just to get to element, set is useless now anyways
        
    def _has_at_least_one_empty_space(self):
        for i in range(len(self.board)):
            if self.board[i] == Space.EMPTY:
                return True
        return False
        
    def start(self):
        """Starts the game
        
        """
        is_player_x_turn = True
        while (self._get_winner() == None) and (self._has_at_least_one_empty_space()):
            first_pass = True # emulating a Do While loop
            first_move_failed = False # so we don't get stuck in an infinite loop
            
            while first_pass or first_move_failed == True:
                first_pass = False
                if is_player_x_turn:
                    attempted_move = self.player_x.query_move(self, first_move_failed)
                else:
                    attempted_move = self.player_o.query_move(self, first_move_failed)

                print(attempted_move)

                try:
                    self._attempt_move(attempted_move)
                    if self.move_writer:
                        self.move_writer.write_move(self.board, is_player_x_turn, attempted_move.index)
                    is_player_x_turn = not is_player_x_turn
                except Board.MoveAlreadyTakenException:
                    print(f"\tmove failed, already taken")
                    first_move_failed = True

        print(self)
        print(f"{self._get_winner()} has won!")
            
    def __str__(self):
        output = ""
        
        for i in range(len(self.board)):
            output += str(self.board[i])
            output += ' '
            
            # prints new lines at end of rows
            if (i!= 0) and (((i + 1) % 3) == 0):
                output += '\n'
        
        return output


# Used this cell to demonstrate a player can play against a random strategy
board = Board(HumanPlayerStrategy(Space.X), RandomPlayerStrategy(Space.O), MoveWritter("train_2.csv"))
board.start()
# Used this cell to write back training data
while True:
    board = Board(HumanPlayerStrategy(Space.X), RandomPlayerStrategy(Space.O), MoveWritter("train_2.csv"))
    board.start()
# Used ths scell to demonstrate a player can play against a decision tree strategy
# 
# Had to convert our `train_2.csv` training set to all floats

# In[115]:


board = Board(DecisionTreeClassifierPlayerStrategy(Space.X), HumanPlayerStrategy(Space.O))
board.start()


# In[ ]:




