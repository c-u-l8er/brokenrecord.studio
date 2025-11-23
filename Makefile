include c_src/Makefile

# Override the target to look in c_src directory
brokenrecord_physics.c: c_src/brokenrecord_physics.c
	@echo "Found c_src/brokenrecord_physics.c"