# Setzt Labels in Templates ein
from random import randrange, shuffle


def class_name(label):
    return label


def listing_parts(label):
    return f"{', '.join(label)}"


def listing_attributes(label):
    # Da es möglich und wahrscheinlich ist, dass das Label insgesamt mehr Token hat,
    # als CLIP als Input nehmen kann (CLIP schneißt alle folgenden Token weg).
    # Werden die Listenelemente zufällig gemischt, damit alle Attribute zu den ersten
    # 76 Token gehören können
    return f"{', '.join(shuffle([' '.join(x) for x in label]))}"


def sentence_class_name(label):
    return f"a photo of a {label}"


def sentence_class_and_random_attribute_label(label):
    """
    Satz mit Klasse und zufällig ausgwählten Attribut

    Beispiel:
    Input: label in Form (Klasse und Attribute):

    ['Black footed Albatross', ['has bill shape', 'hooked seabird'], 
    ['has underparts color', 'buff'], ['has breast pattern', 'multi-colored'], 
    ['has back color', 'buff'], ['has tail shape', 'squared tail'], 
    ['has head pattern', 'plain'], ['has breast color', 'buff'], 
    ['has throat color', 'white'], ['has throat color', 'buff'], 
    ['has eye color', 'black'], ['has eye color', 'buff'], 
    ['has bill length', 'about the same as head'], ['has forehead color', 'buff'], 
    ['has under tail color', 'brown'], ['has under tail color', 'white'], 
    ...

    Output:

    'Black footed Albatross is a bird and has bill color black'
    """

    r_idx = randrange(len(label[1:])) + 1
    return f"{label[0]} is a bird and {' '.join(label[r_idx])}"


def sentence_class_and_part_label(label):
    """
    Satz mit Klasse und Parts (max 15 Parts - 
    maximale Token für CLIP sind 76 - nicht überschreiten)

    Beispiel:
    Input: Label in Form (Klasse und Parts):

    ['Black footed Albatross', 'back', 'beak', 'breast', 'crown', 
    'forehead', 'left wing', 'nape', 'right eye', 'right wing', 
    'tail', 'throat']
    
    Output:

    Black footed Albatross is a bird. The following parts are 
    visible:  back, beak, breast, crown, forehead, left wing, 
    nape, right eye, right wing, tail, throat.
    """

    # r_idx = randrange(len(label[1:])) + 1
    return f"{label[0]} is a bird. The following parts are visible:  {', '.join(label[1:])}"
