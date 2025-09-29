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

#pragma once

#include <stdint.h>


typedef union triplet
{
	struct
	{
		float x;
		float y;
		float z;
	};

	struct
	{
		float w;
		float h;
		float d;
	};
}
triplet_t;


typedef union itriplet
{
	struct
	{
		int x;
		int y;
		int z;
	};

	struct
	{
		int w;
		int h;
		int d;
	};
}
itriplet_t;


typedef union half_extent_3d
{
	struct
	{
		union
		{
			triplet_t pos;

			struct
			{
				float x;
				float y;
				float z;
			};
		};

		union
		{
			triplet_t size;

			struct
			{
				float w;
				float h;
				float d;
			};
		};
	};

	struct
	{
		float top;
		float left;
		float right;
		float bottom;
		float front;
		float back;
	};
}
half_extent_3d_t;


typedef struct rect_extent_3d
{
	union
	{
		triplet_t min;

		struct
		{
			float min_x;
			float min_y;
			float min_z;
		};
	};

	union
	{
		triplet_t max;

		struct
		{
			float max_x;
			float max_y;
			float max_z;
		};
	};
}
rect_extent_3d_t;


extern float
triplet_dot(
	triplet_t a,
	triplet_t b
	);


extern float
triplet_length(
	triplet_t v
	);


extern triplet_t
triplet_normalize(
	triplet_t v
	);


extern triplet_t
triplet_scale(
	triplet_t v,
	float s
	);


extern triplet_t
triplet_negate(
	triplet_t v
	);


extern triplet_t
triplet_add(
	triplet_t a,
	triplet_t b
	);


extern triplet_t
triplet_sub(
	triplet_t a,
	triplet_t b
	);


extern triplet_t
triplet_cross(
	triplet_t a,
	triplet_t b
	);


extern bool
rect_extent_3d_intersects(
	rect_extent_3d_t a,
	rect_extent_3d_t b
	);


extern bool
rect_extent_3d_is_inside(
	rect_extent_3d_t a,
	rect_extent_3d_t b
	);


extern rect_extent_3d_t
half_to_rect_3d_extent(
	half_extent_3d_t extent
	);


extern half_extent_3d_t
rect_to_half_3d_extent(
	rect_extent_3d_t extent
	);
