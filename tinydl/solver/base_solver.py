class BaseSolver():
    def __init__(self) -> None:
        pass
    
    '''
    This method should implement the logic for a single optimization step.
    Include calculate the gradients, update the parameters, and clear the gradients.
    '''
    def step(self) -> None:
        raise NotImplementedError
    
    '''
    This method should implement the logic for updating the parameters of the model.
    '''
    def update_parameters(self) -> None:
        raise NotImplementedError
