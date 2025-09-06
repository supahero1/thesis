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


SHADERS := $(filter-out $(shell grep -l '^/\* skip \*/' shaders/*),$(wildcard shaders/*))
BIN_SHADERS=$(SHADERS:shaders/%=bin/shaders/%.spv)

bin/shaders/%.vert.spv: shaders/%.vert | bin/shaders/ thesis/shaders/
	glslc -O -fshader-stage=vert $< -o $@
	$(CP) $@ thesis/shaders/

bin/shaders/%.frag.spv: shaders/%.frag | bin/shaders/ thesis/shaders/
	glslc -O -fshader-stage=frag $< -o $@
	$(CP) $@ thesis/shaders/

bin/shaders/ssao_blur_h.frag.spv: shaders/ssao_blur.frag
bin/shaders/ssao_blur_v.frag.spv: shaders/ssao_blur.frag

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
	\
	args="--xr_runtime=$$xr_runtime"; \
	[[ -n "$(XR_ENABLE)" ]] && args="$$args --xr_enable=$(XR_ENABLE)"; \
	[[ -n "$(WINDOW_ENABLE)" ]] && args="$$args --window_enable=$(WINDOW_ENABLE)"; \
	[[ -n "$(WINDOW_FULLSCREEN)" ]] && args="$$args --window_fullscreen=$(WINDOW_FULLSCREEN)"; \
	[[ -n "$(WINDOW_WIDTH)" ]] && args="$$args --window_width=$(WINDOW_WIDTH)"; \
	[[ -n "$(WINDOW_HEIGHT)" ]] && args="$$args --window_height=$(WINDOW_HEIGHT)"; \
	[[ -n "$(VK_MAX_MSAA_SAMPLES)" ]] && args="$$args --vk_max_msaa_samples=$(VK_MAX_MSAA_SAMPLES)"; \
	[[ -n "$(VK_SAMPLE_SHADING)" ]] && args="$$args --vk_sample_shading=$(VK_SAMPLE_SHADING)"; \
	[[ -n "$(VK_MIN_SAMPLE_SHADING)" ]] && args="$$args --vk_min_sample_shading=$(VK_MIN_SAMPLE_SHADING)"; \
	[[ -n "$(VK_MIPMAP_LEVELS)" ]] && args="$$args --vk_mipmap_levels=$(VK_MIPMAP_LEVELS)"; \
	[[ -n "$(VK_MAX_ANISOTROPY)" ]] && args="$$args --vk_max_anisotropy=$(VK_MAX_ANISOTROPY)"; \
	[[ -n "$(VK_PREVIEW)" ]] && args="$$args --vk_preview=$(VK_PREVIEW)"; \
	[[ -n "$(VK_SHADOW_MAP_SIZE)" ]] && args="$$args --vk_shadow_map_size=$(VK_SHADOW_MAP_SIZE)"; \
	[[ -n "$(VK_ENABLE_DEPTH_SHADOWS)" ]] && args="$$args --vk_enable_depth_shadows=$(VK_ENABLE_DEPTH_SHADOWS)"; \
	[[ -n "$(VK_ENABLE_BACKFACE_SHADOWS)" ]] && args="$$args --vk_enable_backface_shadows=$(VK_ENABLE_BACKFACE_SHADOWS)"; \
	[[ -n "$(VK_ENABLE_SPECULAR)" ]] && args="$$args --vk_enable_specular=$(VK_ENABLE_SPECULAR)"; \
	[[ -n "$(VK_SHADOW_VALUE)" ]] && args="$$args --vk_shadow_value=$(VK_SHADOW_VALUE)"; \
	[[ -n "$(VK_LAMBERT_START_ANGLE)" ]] && args="$$args --vk_lambert_start_angle=$(VK_LAMBERT_START_ANGLE)"; \
	[[ -n "$(VK_ENABLE_SSAO)" ]] && args="$$args --vk_enable_ssao=$(VK_ENABLE_SSAO)"; \
	[[ -n "$(VK_SSAO_KERNEL_SIZE)" ]] && args="$$args --vk_ssao_kernel_size=$(VK_SSAO_KERNEL_SIZE)"; \
	[[ -n "$(VK_SSAO_NOISE_SIZE)" ]] && args="$$args --vk_ssao_noise_size=$(VK_SSAO_NOISE_SIZE)"; \
	[[ -n "$(VK_SSAO_RADIUS)" ]] && args="$$args --vk_ssao_radius=$(VK_SSAO_RADIUS)"; \
	[[ -n "$(VK_SSAO_BIAS)" ]] && args="$$args --vk_ssao_bias=$(VK_SSAO_BIAS)"; \
	[[ -n "$(VK_SSAO_POWER)" ]] && args="$$args --vk_ssao_power=$(VK_SSAO_POWER)"; \
	[[ -n "$(VK_SSAO_DEPTH_K)" ]] && args="$$args --vk_ssao_depth_k=$(VK_SSAO_DEPTH_K)"; \
	[[ -n "$(VK_SSAO_DEPTH_GAMMA)" ]] && args="$$args --vk_ssao_depth_gamma=$(VK_SSAO_DEPTH_GAMMA)"; \
	[[ -n "$(VK_SSAO_DEBUG)" ]] && args="$$args --vk_ssao_debug=$(VK_SSAO_DEBUG)"; \
	[[ -n "$(VK_SSAO_SCALE)" ]] && args="$$args --vk_ssao_scale=$(VK_SSAO_SCALE)"; \
	[[ -n "$(VK_SSAO_BLUR_RADIUS)" ]] && args="$$args --vk_ssao_blur_radius=$(VK_SSAO_BLUR_RADIUS)"; \
	[[ -n "$(VK_SSAO_BLUR_FALLOFF)" ]] && args="$$args --vk_ssao_blur_falloff=$(VK_SSAO_BLUR_FALLOFF)"; \
	[[ -n "$(VK_SSAO_BLUR_DEPTH_TOLERANCE)" ]] && args="$$args --vk_ssao_blur_depth_tolerance=$(VK_SSAO_BLUR_DEPTH_TOLERANCE)"; \
	\
	cd thesis; $(VALGRIND_CALL) ./thesis_app $$args
