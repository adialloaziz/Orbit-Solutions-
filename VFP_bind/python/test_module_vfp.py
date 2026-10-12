


from module_vfp import integ_vfp
t1 = 0.0
t2 = 20.
nt = 800
ntint=10
sol = integ_vfp(t1, t2 , nt, ntint )
print("Type of the solution:  ", type(sol.y))
print("Mass ini and mass fin: ", sol.mass[0], sol.mass[-1])
print("density length", len(sol.density))