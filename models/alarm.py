"""
Alarm system for training hooks.
Provides different types of alarms that trigger based on training progress.
"""

class BaseAlarm:
    """Base class for all alarm types."""
    
    def __init__(self, logger, target='step', **kwargs):
        """
        Parameters
        ----------
        logger : logging.Logger
            Logger instance
        target : str
            Target key in batch to monitor (e.g., 'step', 'epoch')
        """
        self.logger = logger
        self.target = target
    
    def __call__(self, batch):
        """
        Check if alarm should trigger.
        
        Parameters
        ----------
        batch : dict
            Training batch containing step/epoch information
            
        Returns
        -------
        bool
            True if alarm should trigger, False otherwise
        """
        raise NotImplementedError


class SilentAlarm(BaseAlarm):
    """Alarm that never triggers - used as default."""
    
    def __call__(self, batch):
        return False


class CountAlarm(BaseAlarm):
    """Alarm that triggers every N steps/epochs."""
    
    def __init__(self, logger, target='step', step=1000, start=0, **kwargs):
        """
        Parameters
        ----------
        step : int
            Interval between triggers
        start : int
            First step/epoch to start counting from
        """
        super().__init__(logger, target, **kwargs)
        self.step = step
        self.start = start
        
    def __call__(self, batch):
        if self.target not in batch:
            return False
        
        current = batch[self.target]
        if current < self.start:
            return False
        
        return (current - self.start) % self.step == 0


class ListAlarm(BaseAlarm):
    """Alarm that triggers at specific steps/epochs."""
    
    def __init__(self, logger, target='step', list=None, **kwargs):
        """
        Parameters
        ----------
        list : list
            List of specific steps/epochs to trigger at
        """
        super().__init__(logger, target, **kwargs)
        self.trigger_list = set(list or [])
        
    def __call__(self, batch):
        if self.target not in batch:
            return False
        
        return batch[self.target] in self.trigger_list


class ThresholdAlarm(BaseAlarm):
    """Alarm that triggers when value exceeds threshold."""
    
    def __init__(self, logger, target='step', threshold=float('inf'), **kwargs):
        """
        Parameters
        ----------
        threshold : float
            Threshold value to trigger at
        """
        super().__init__(logger, target, **kwargs)
        self.threshold = threshold
        
    def __call__(self, batch):
        if self.target not in batch:
            return False
        
        return batch[self.target] >= self.threshold


# Registry of alarm types
alarm_type2class = {
    'silent': SilentAlarm,
    'count': CountAlarm,
    'list': ListAlarm,
    'threshold': ThresholdAlarm,
}


def get_alarm(logger, type='silent', **kwargs):
    """
    Factory function to create alarm instances.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    type : str
        Type of alarm ('silent', 'count', 'list', 'threshold')
    **kwargs
        Additional parameters for the specific alarm type
        
    Returns
    -------
    BaseAlarm
        Alarm instance that can be called with batch to check if it should trigger
        
    Examples
    --------
    >>> # Silent alarm (never triggers)
    >>> alarm = get_alarm(logger, type='silent')
    
    >>> # Count alarm (triggers every 1000 steps)
    >>> alarm = get_alarm(logger, type='count', target='step', step=1000)
    
    >>> # List alarm (triggers at specific steps)
    >>> alarm = get_alarm(logger, type='list', target='step', list=[100, 500, 1000])
    
    >>> # Check if alarm triggers
    >>> batch = {'step': 1000, 'epoch': 1}
    >>> if alarm(batch):
    >>>     print("Alarm triggered!")
    """
    if type not in alarm_type2class:
        logger.warning(f"Unknown alarm type '{type}', using 'silent' instead")
        type = 'silent'
    
    alarm_class = alarm_type2class[type]
    return alarm_class(logger=logger, **kwargs)