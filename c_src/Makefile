ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

CFLAGS = -O3 -march=native -fPIC -std=c11
CFLAGS += -I$(ERLANG_PATH)
CFLAGS += -mavx2 -mfma -fopenmp
CFLAGS += -Wall -Wextra

LDFLAGS = -shared -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
    LDFLAGS += -dynamiclib -undefined dynamic_lookup
endif

PRIV_DIR = ../priv
NIF_SO = $(PRIV_DIR)/brokenrecord_physics.so

.PHONY: all clean

all: $(NIF_SO)

$(PRIV_DIR):
	mkdir -p $(PRIV_DIR)

$(NIF_SO): $(PRIV_DIR) brokenrecord_physics.c c_src/brokenrecord_physics.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ c_src/brokenrecord_physics.c

clean:
	rm -f $(NIF_SO)