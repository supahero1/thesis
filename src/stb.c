/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thesis/debug.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>


void
stbi_print_failure(
	void
	)
{
	const char* reason = stbi_failure_reason();
	fprintf(stderr, "%s\n", reason ? "stb: unknown error\n" : reason);
}


void
stbi_flip_horizontally(
	uint32_t* data,
	uint32_t width,
	uint32_t height
	)
{
	assert_ptr(data, width * height * 4);

	for(uint32_t y = 0; y < height; ++y)
	{
		uint32_t* row_start = data + y * width;
		uint32_t* row_end = row_start + width - 1;

		while(row_start < row_end)
		{
			uint32_t temp = *row_start;
			*row_start++ = *row_end;
			*row_end-- = temp;
		}
	}
}
