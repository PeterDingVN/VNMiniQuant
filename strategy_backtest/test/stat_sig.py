from config import NUM_OF_PERM

class StatTest:

    """
        The class is used to do statistical test with quasi_p value for the Strategy
        Strategy is a bunch of signals created by a trained algorithm (the algorithm is sw else, optimizing on errors
        such as MSE, RMSE if we are doing ML => after this the ML is used to create signal => this signal is the Strategy
        => And our stat test here is testing if such Strategy, or more precisely THE PROCESS FROM TRAINING TO GENERATING
        SIGNALS AS A WHOLE, is NOT good by chance/luck
    
    Returns:
        A quasi p-value
    """
    
    @staticmethod
    def quasi_pvalue(mcpt_better: list | int, all_trials: int=NUM_OF_PERM):

        if isinstance(mcpt_better, list):
            betterperf = len(mcpt_better)
        else:
            betterperf = mcpt_better

        quasi_p = betterperf / all_trials

        return quasi_p