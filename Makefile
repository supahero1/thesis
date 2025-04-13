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

	kill -0 "$$(cat /run/user/1000/monado.pid)" 2>/dev/null; \
	runtime=; \
	if [[ "$$?" -eq 0 ]]; then \
		$(CP) /etc/openxr/1/monado_runtime.json ~/.config/openxr/1/active_runtime.json; \
		runtime=monado; \
	else \
		$(CP) /etc/openxr/1/wivrn_runtime.json ~/.config/openxr/1/active_runtime.json; \
		runtime=wivrn; \
	fi; \
	cd thesis; $(VALGRIND_CALL) ./app --runtime=$$runtime
