(define(domain pacman))
(:predicates 
(is_empty ?position)
(pacman_at ?position)
(food_at ?position)
(capsule_at ?position)
(ghost_at ?position)
(adjacent ?pos1 ?pos2)
(SuperPacman)
)


(:action move
:paramaters(?pos1 ?pos2)
:precondition(and(pacman_at ?pos1) (is_empty ?pos2) (adjacent ?pos1 ?pos2))
:effect(and(pacman_at ?pos2) (is_empty ?pos1))

)
(:action eatfood
:paramaters(?pos1 ?pos2)
:precondition(and(pacman_at ?pos1) (food_at ?pos2) (adjacent ?pos1 ?pos2)not(is_empty ?pos2))
:effect(and(pacman_at?pos2)(is_empty ?pos1))

)
(:action eatcapsule
:paramaters(?pos1 ?pos2)
:precondition(and(pacman_at ?pos1) (capsule_at ?pos2) (adjacent ?pos1 ?pos2)not(is_empty ?pos2))
:effect(and(pacman_at?pos2)(is_empty ?pos1)(SuperPacman))

)

(:action eatghost
:paramaters(?pos1 ?pos2)
:precondition(and(SuperPacman)(pacman_at ?pos1) (ghost_at ?pos2) (adjacent ?pos1 ?pos2)not(is_empty ?pos2))
:effect(and(pacman_at?pos2)(is_empty ?pos1)

)



