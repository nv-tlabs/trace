from typing import Union

from tbsim.dynamics.unicycle import Unicycle
from tbsim.dynamics.base import DynType

from tbsim.dynamics.base import Dynamics, DynType, forward_dynamics

def get_dynamics_model(dyn_type: Union[str, DynType]):
    if dyn_type in ["Unicycle", DynType.UNICYCLE]:
        return Unicycle
    else:
        raise NotImplementedError("Dynamics model {} is not implemented".format(dyn_type))
