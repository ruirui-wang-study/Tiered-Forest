class BaseAgent:
    def __init__(self, kg, monitor):
        self.kg = kg
        self.monitor = monitor
    
    def answer(self, question):
        raise NotImplementedError

class TieredForestAgent(BaseAgent):
    def __init__(self, kg, monitor, t_drop=0.2, t_pass=0.8):
        super().__init__(kg, monitor)
        self.tier1 = Tier1Pruner()
        self.tier2 = Tier2Ranker()
        self.tier3 = Tier3Reasoner(monitor) # 传入 monitor 记账
        self.params = (t_drop, t_pass)

    def answer(self, question):
        # 1. Start Search
        # 2. Tier 1 Pruning
        # 3. Tier 2 Scoring
        # 4. Dual-Threshold Logic (Drop/Pass/Escalate)
        # 5. Tier 3 Verification (if escalated)
        pass