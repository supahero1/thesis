#   Copyright 2025 Franciszek Balcerak
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

RM := rm -f
CP := cp
MV := mv
SHELL := bash


ifeq ($(VALGRIND),1)
VALGRIND_CALL := valgrind --leak-check=full --show-leak-kinds=all \
	--suppressions=../val_sup.txt --log-file="val_log.txt" --
endif

ifeq ($(KCACHEGRIND),1)
VALGRIND_CALL := valgrind --tool=callgrind --
endif


.PHONY: all
all:
	@printf "Specify one (or more) of the following:\n\
	\n\
	app         builds the app\n\
	clean       removes any built executables\n\
	\n\
	Specify RELEASE=1 for a production build\n\
	Specify RELEASE=2 for a native build (faster than production but not portable)\n"


bin/shaders/ thesis/shaders/:
	mkdir -p $@

.PHONY: clean
clean:
	$(RM) -r bin/


bin/shaders/%.spv: shaders/%.glsl | bin/shaders/ thesis/shaders/
	glslc -O -fshader-stage=$* $< -o $@
	$(CP) $@ thesis/shaders/

.PHONY: shaders
shaders: bin/shaders/vert.spv bin/shaders/frag.spv


.PHONY: app
app: shaders
	scons app -j $(shell nproc)

	if [[ ! -d thesis/shaders/ ]]; then \
		$(CP) -r bin/shaders/ thesis/; \
	fi

	$(CP) bin/app thesis/

ifeq ($(OS),Windows_NT)
	if [[ ! -f thesis/SDL3.dll ]]; then \
		$(CP) "C:\\msys64\\mingw64\\bin\\SDL3.dll" thesis/; \
	fi
	if [[ ! -f thesis/libwinpthread-1.dll ]]; then \
		$(CP) "C:\\msys64\\mingw64\\bin\\libwinpthread-1.dll" thesis/; \
	fi
else
	if [[ ! -f thesis/libvulkan.so.1 ]]; then \
		$(CP) \
			$$(ls -1 /usr/lib/x86_64-linux-gnu/libvulkan.so.1.4.* | sort -V | tail -n 1) \
			thesis/libvulkan.so.1; \
	fi
	if ! ls thesis/libSDL3.so.* 1>/dev/null 2>&1; then \
		$(CP) \
			$$(ls -1 /usr/local/lib/libSDL3.so.* | sort -V | tail -n 1) \
			thesis/; \
	fi
endif

	cd thesis; $(VALGRIND_CALL) ./app
