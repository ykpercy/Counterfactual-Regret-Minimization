import numpy as np


class LiarDieTrainer:
    """Liar Die definitions"""
    DOUBT = 0
    ACCEPT = 1
    # NUM_ACTIONS = 2
    # rng = np.random.default_rng(12345)
    # rintger = rng.integers(low=0, high=self.sides, size=3)

    def __init__(self, sides):
        self.sides = sides
        self.responseNodes = np.zeros((self.sides + 1, self.sides + 1), dtype=float)
        self.claimNodes = np.zeros((self.sides + 1, self.sides + 1), dtype=float)

    def random_int(self):
        rng = np.random.default_rng(12345)
        randomint = rng.integers(low=0, high=self.sides, size=1)
        return randomint

    class Node:
        """Liar Die player decision node"""
        def __init__(self, pPlayer, pOpponent):
            """Liar Die node definitions. """
            # self.NUM_ACTIONS = LiarDieTrainer.NUM_ACTIONS
            self.regretSum = np.zeros(1, dtype=float)
            self.strategy = np.zeros(1, dtype=float)
            self.strategySum = np.zeros(1, dtype=float)
            self.pPlayer = pPlayer
            self.pOpponent = pOpponent

        def Node(self, numActions):
            """Liar Die node constructor"""
            self.regretSum = [numActions]
            self.strategy = [numActions]
            self.strategySum = [numActions]

        def getStrategy(self):
            """
            Get Liar Die node current mixed strategy through regret-matching
            """
            normalizingSum = 0
            for a in range(len(self.strategy)):
                self.strategy[a] = max(self.regretSum[a], 0)
                normalizingSum += self.strategy[a]

            for a in range(len(self.strategy)):
                if normalizingSum > 0:
                    self.strategy[a] /= normalizingSum
                else:
                    self.strategy[a] = 1.0 / len(self.strategy)

            for a in range(len(self.strategy)):
                self.strategySum[a] += self.pPlayer * self.strategy[a]

            return self.strategy

        def getAverageStrategy(self):
            """
            Get Liar Die node average mixed strategy
            """
            normalizingSum = 0
            for a in range(len(self.strategySum)):
                normalizingSum += self.strategySum[a]

            for a in range(len(self.strategySum)):
                if normalizingSum > 0:
                    self.strategySum[a] /= normalizingSum
                else:
                    self.strategySum[a] = 1.0 / len(self.strategySum)

            return self.strategySum

    def LiarDieTrainer(self, sides):
        """
        Construct trainer and allocate player decision nodes
        """
        self.sides = sides
        # responseNodes = Node[sides][sides + 1]  # Node need to be replaced by python data structure
        responseNodes = np.zeros((self.sides, self.sides + 1), dtype=float)
        for myClaim in range(self.sides+1):
            for oppClaim in range(1, self.sides+1):
                if oppClaim == 0 or oppClaim == self.sides:
                    responseNodes[myClaim][oppClaim] = 1
                else:
                    responseNodes[myClaim][oppClaim] = 2

        # claimNodes = Node[sides][sides + 1]  # Node need to be replaced by python data structure
        claimNodes = np.zeros((self.sides, self.sides + 1), dtype=float)
        for oppClaim in range(self.sides):
            for roll in range(1, self.sides+1):
                claimNodes[oppClaim][roll] = Node(sides - oppClaim)  # Node need to be replaced by python data structure

    def train(self, iterations):
        regret = np.zeros(self.sides, dtype=float)
        rollAfterAcceptingClaim = np.zeros(self.sides, dtype=float)
        for _ in range(iterations):
            """Initialize rolls and starting probabilities"""
            for i in range(len(rollAfterAcceptingClaim)):
                rollAfterAcceptingClaim[i] = self.random_int().nextInt(self.sides) + 1  # node to be replaced
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pPlayer = 1
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pOpponent = 1

            """Accumulate realization weights forward"""
            for oppClaim in range(self.sides+1):
                """Visit response nodes forward"""
                if oppClaim > 0:
                    for myClaim in range(oppClaim):
                        node = self.responseNodes[myClaim][oppClaim]
                        actionProb = [node.getStrategy()]
                        if oppClaim < self.sides:
                            nextNode = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                            nextNode.pPlayer += actionProb[1] * node.pPlayer
                            nextNode.pOpponent += node.pOpponent

                """Visit claim nodes forward"""
                if oppClaim < self.sides:
                    node = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                    actionProb = [node.getStrategy()]
                    for myClaim in range(1, self.sides+1):
                        nextClaimProb = actionProb[myClaim - oppClaim - 1]
                        if nextClaimProb > 0:
                            nextNode = self.responseNodes[oppClaim][myClaim]
                            nextNode.pPlayer += node.pOpponent
                            nextNode.pOpponent += nextClaimProb * node.pPlayer

            """Backpropagate utilities, adjusting regrets and strategies"""
            for oppClaim in range(self.sides, -1, -1):
                """Visit claim nodes backward"""
                if oppClaim < self.sides:
                    node = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                    actionProb = [node.getStrategy()]
                    node.u = 0.0
                    for myClaim in range(oppClaim+1, self.sides+1):
                        actionIndex = myClaim - oppClaim - 1
                        nextNode = self.responseNodes[oppClaim][myClaim]
                        childUtil = - nextNode.u
                        regret[actionIndex] = childUtil
                        node.u += actionProb[actionIndex] * childUtil

                    for a in range(len(actionProb)):
                        regret[a] -= node.u
                        node.regretSum[a] += node.pOpponent * regret[a]

                    node.pPlayer = node.pOpponent = 0

                """Visit response nodes backward"""
                if oppClaim > 0:
                    for myClaim in range(oppClaim):
                        node = self.responseNodes[myClaim][oppClaim]
                        actionProb = [node.getStrategy()]
                        node.u = 0.0
                        doubtUtil = 1 if oppClaim > rollAfterAcceptingClaim[myClaim] else -1
                        regret[LiarDieTrainer.DOUBT] = doubtUtil
                        node.u += actionProb[LiarDieTrainer.DOUBT] * doubtUtil
                        if oppClaim < self.sides:
                            nextNode = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                            regret[LiarDieTrainer.ACCEPT] = nextNode.u
                            node.u += actionProb[LiarDieTrainer.ACCEPT] * nextNode.u
                        for a in range(len(actionProb)):
                            regret[a] -= node.u
                            node.regretSum[a] += node.pOpponent * regret[a]

                        node.pPlayer = node.pOpponent = 0

            """Reset strategy sums after half of training"""
            if iter == iterations / 2:
                for nodes in self.responseNodes:
                    for node in nodes:
                        if node is not None:
                            for a in range(len(node.strategySum)):
                                node.strategySum[a] = 0

                for nodes in self.claimNodes:
                    for node in nodes:
                        if node is not None:
                            for a in range(len(node.strategySum)):
                                node.strategySum[a] = 0

        """Print resulting strategy"""
        for initialRoll in range(1, self.sides+1):
            print(f"Initial claim policy with roll {initialRoll}")
            for prob in self.claimNodes[0][initialRoll].getAverageStrategy():
                print(f"{prob}\n")

        print(f"\nOld Claim\tNew Claim\tAction Probabilities")

        for myClaim in range(self.sides+1):
            for oppClaim in range(myClaim+1, self.sides+1):
                print(f"\t{myClaim}\t{oppClaim}\t{''.join(self.responseNodes[myClaim][oppClaim].getAverageStrategy())}")

        print(f"\nOld Claim\tRoll\tAction Probabilities")

        for oppClaim in range(self.sides):
            for roll in range(1, self.sides+1):
                print(f"{oppClaim}\t{roll}\t{''.join(self.claimNodes[oppClaim][roll].getAverageStrategy())}")

    def main(self, *args):
        """LiarDieTrainer main method"""
        trainer = LiarDieTrainer(sides=6)
        trainer.train(1000000)

































