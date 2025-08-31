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

#include <thesis/extent_3d.h>


bool
rect_extent_3d_intersects(
	rect_extent_3d_t a,
	rect_extent_3d_t b
	)
{
	return
		a.max_x >= b.min_x &&
		a.max_y >= b.min_y &&
		a.max_z >= b.min_z &&
		b.max_x >= a.min_x &&
		b.max_y >= a.min_y &&
		b.max_z >= a.min_z;
}


bool
rect_extent_3d_is_inside(
	rect_extent_3d_t a,
	rect_extent_3d_t b
	)
{
	return
		a.min_x > b.min_x &&
		a.min_y > b.min_y &&
		a.min_z > b.min_z &&
		b.max_x > a.max_x &&
		b.max_y > a.max_y &&
		b.max_z > a.max_z;
}


rect_extent_3d_t
half_to_rect_3d_extent(
	half_extent_3d_t extent
	)
{
	return
	(rect_extent_3d_t)
	{
		.min_x = extent.x - extent.w,
		.min_y = extent.y - extent.h,
		.min_z = extent.z - extent.d,
		.max_x = extent.x + extent.w,
		.max_y = extent.y + extent.h,
		.max_z = extent.z + extent.d
	};
}


half_extent_3d_t
rect_to_half_3d_extent(
	rect_extent_3d_t extent
	)
{
	return
	(half_extent_3d_t)
	{
		.x = (extent.max_x + extent.min_x) * 0.5f,
		.y = (extent.max_y + extent.min_y) * 0.5f,
		.z = (extent.max_z + extent.min_z) * 0.5f,
		.w = (extent.max_x - extent.min_x) * 0.5f,
		.h = (extent.max_y - extent.min_y) * 0.5f,
		.d = (extent.max_z - extent.min_z) * 0.5f
	};
}
