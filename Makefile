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
	--suppressions=../val_sup.txt --log-file="val_log.txt" --track-origins=yes --
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
	$(RM) -r bin/ thesis/


SHADERS=$(wildcard shaders/*)
BIN_SHADERS=$(SHADERS:shaders/%=bin/shaders/%.spv)

bin/shaders/%.vert.spv: shaders/%.vert | bin/shaders/ thesis/shaders/
	glslc -O -fshader-stage=vert $< -o $@
	$(CP) $@ thesis/shaders/

bin/shaders/%.frag.spv: shaders/%.frag | bin/shaders/ thesis/shaders/
	glslc -O -fshader-stage=frag $< -o $@
	$(CP) $@ thesis/shaders/

.PHONY: shaders
shaders: $(BIN_SHADERS)


.PHONY: app
app: shaders
	scons app -j $(shell nproc)

	if [[ ! -d thesis/assets/ ]]; then \
		$(CP) -r assets/ thesis/assets/; \
	fi

	if [[ ! -d thesis/shaders/ ]]; then \
		$(CP) -r bin/shaders/ thesis/shaders/; \
	fi

	$(CP) bin/thesis_app thesis/

	pid_file=/run/user/1000/monado.pid; \
	if [[ ! -e "$$pid_file" ]] || ! kill -0 "$$(cat $$pid_file 2>/dev/null)" 2>/dev/null; then \
		$(CP) /etc/openxr/1/monado_runtime.json ~/.config/openxr/1/active_runtime.json; \
		xr_runtime=monado; \
	else \
		$(CP) /etc/openxr/1/wivrn_runtime.json ~/.config/openxr/1/active_runtime.json; \
		xr_runtime=wivrn; \
	fi; \
	cd thesis; $(VALGRIND_CALL) ./thesis_app \
		--xr_runtime=$$xr_runtime \
		--window_width=$(WINDOW_WIDTH) \
		--window_height=$(WINDOW_HEIGHT) \
		--vk_sample_shading=$(VK_SAMPLE_SHADING) \
		--vk_mipmap_levels=$(VK_MIPMAP_LEVELS) \
		--vk_anisotropy=$(VK_ANISOTROPY) \
		--vk_preview=$(VK_PREVIEW) \
		--vk_shadow_map_size=$(VK_SHADOW_MAP_SIZE) \
		--vk_enable_depth_shadows=$(VK_ENABLE_DEPTH_SHADOWS) \
		--vk_enable_backface_shadows=$(VK_ENABLE_BACKFACE_SHADOWS) \
		--vk_enable_specular=$(VK_ENABLE_SPECULAR) \
		--vk_shadow_value=$(VK_SHADOW_VALUE) \
		--vk_lambert_start_angle=$(VK_LAMBERT_START_ANGLE) \
		--vk_enable_ssao=$(VK_ENABLE_SSAO) \
		--vk_ssao_kernel_size=$(VK_SSAO_KERNEL_SIZE) \
		--vk_ssao_radius=$(VK_SSAO_RADIUS)

