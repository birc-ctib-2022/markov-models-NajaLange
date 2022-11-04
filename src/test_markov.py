"""Testing Markov models."""

from markov import (
    MarkovModel,
    likelihood,
    likelihood_log
)
import math

SUNNY = 0
CLOUDY = 1

X = [SUNNY, SUNNY, CLOUDY, SUNNY]

init_probs = [0.1, 0.9]  # it is almost always cloudy
transitions_from_SUNNY = [0.3, 0.7]
transitions_from_CLOUDY = [0.4, 0.6]
transition_probs = [
    transitions_from_SUNNY,
    transitions_from_CLOUDY
]
    
M = MarkovModel(init_probs, transition_probs)

def test_likelihood() -> None:
    """Test your code."""

    result = likelihood(X, M)
    exp = 0.1*0.3*0.7*0.4

    assert exp == result 

def test_likelihood_log() -> None:
    """Test your code."""

    result = likelihood_log(X, M)
    exp = math.log(0.1)*math.log(0.3)*math.log(0.7)*math.log(0.4)

    assert exp == result 

print(likelihood_log(X, M))