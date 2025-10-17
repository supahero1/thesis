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

#include <thesis/time.h>
#include <thesis/collider.h>
#include <thesis/alloc_ext.h>

#include <thesis/octree.h>

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#define COLLIDER_MAP_SIZE 1000000.0f
#define COLLIDER_HAND_SIZE 2.0f
#define COLLIDER_HAND_V_COUNT 10
#define COLLIDER_HAND_V_MULTIPLIER 2.0f
#define COLLIDER_STATS_SIZE 64
#define COLLIDER_RESTITUTION 0.8f
#define COLLISION_IMPULSE_FACTOR -(1.0f + COLLIDER_RESTITUTION)
#define COLLIDER_VELOCITY_LIMIT 15.0f
#define COLLIDER_FRICTION 0.4f
#define COLLIDER_ANGULAR_DAMPING 0.995f
#define COLLIDER_LOW_SPEED_THRESHOLD 0.5f
#define COLLIDER_LOW_ANGULAR_THRESHOLD 0.1f
#define COLLIDER_LOW_SPEED_DAMPING 0.99f
#define COLLIDER_ROLL_ALIGN_FACTOR 0.1f
#define COLLIDER_MAGNUS_FACTOR 0.001f


struct collider
{
	stats_t stats;
	octree_t octree;
	float delta;

	collider_entity_t* balls;
	uint32_t balls_used;
	uint32_t balls_size;

	collider_entity_t** query_entities;
	uint32_t query_entities_used;
	uint32_t query_entities_size;

	float ball_mass;
	float inv_ball_mass;
	float ball_radius;
	float ball_diameter;
	float ball_moment_inertia;
	float inv_ball_moment_inertia;

	float hand_radius;
	triplet_t hand_pos;
	triplet_t hand_vs[COLLIDER_HAND_V_COUNT];
	triplet_t hand_v;
	uint32_t hand_v_count;
	uint32_t hand_v_index;
	bool fist;
	bool grab;
};


collider_t
collider_init(
	stats_t stats
	)
{
	collider_t collider = alloc_calloc(collider, 1);
	assert_not_null(collider);

	collider->stats = stats;
	stats_add(collider->stats, "collider_normalize", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_collide", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_resolve", COLLIDER_STATS_SIZE);

	collider->octree.half_extent =
	(half_extent_3d_t)
	{
		.w = COLLIDER_MAP_SIZE,
		.h = COLLIDER_MAP_SIZE,
		.d = COLLIDER_MAP_SIZE,
	};
	collider->octree.rect_extent = half_to_rect_3d_extent(collider->octree.half_extent);
	collider->octree.min_size = 32.0f;
	octree_init(&collider->octree);

	collider->ball_mass = 1.0f;
	collider->inv_ball_mass = 1.0f / collider->ball_mass;

	return collider;
}


void
collider_free(
	collider_t collider
	)
{
	assert_not_null(collider);

	alloc_free(collider->query_entities, collider->query_entities_size);

	alloc_free(collider->balls, collider->balls_size);

	stats_del(collider->stats, "collider_resolve");
	stats_del(collider->stats, "collider_collide");
	stats_del(collider->stats, "collider_normalize");

	octree_free(&collider->octree);

	alloc_free(collider, 1);
}


float
collider_get_scale(
	collider_t collider
	)
{
	assert_not_null(collider);

	return collider->ball_radius / 0.1213f;
}


void
collider_add(
	collider_t collider,
	const collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	if(entity->type == COLLIDER_ENTITY_TYPE_TRIANGLE)
	{
		octree_insert(&collider->octree, entity);
		return;
	}

	if(collider->balls_used >= collider->balls_size)
	{
		uint32_t new_size = (collider->balls_size << 1) | 1;

		collider->balls = alloc_remalloc(collider->balls, collider->balls_size, new_size);
		assert_not_null(collider->balls);

		collider->balls_size = new_size;
	}

	collider->balls[collider->balls_used++] = *entity;

	if(entity->type == COLLIDER_ENTITY_TYPE_FIST)
	{
		return;
	}

	if(collider->ball_radius == 0.0f)
	{
		collider->ball_radius = (entity->rect_extent.max.x - entity->rect_extent.min.x) * 0.5f;
		collider->ball_diameter = collider->ball_radius * 2.0f;
		collider->ball_moment_inertia = 0.4f * collider->ball_mass * collider->ball_radius * collider->ball_radius;
		collider->inv_ball_moment_inertia = 1.0f / collider->ball_moment_inertia;

		collider->hand_radius = collider->ball_radius * COLLIDER_HAND_SIZE;

		printf("Basketball radius: %.02f\n", collider->ball_radius);
	}

	assert_lt(collider->ball_radius - (entity->rect_extent.max.x - entity->rect_extent.min.x) * 0.5f, 0.01f);
	assert_lt(collider->ball_radius - (entity->rect_extent.max.y - entity->rect_extent.min.y) * 0.5f, 0.01f);
	assert_lt(collider->ball_radius - (entity->rect_extent.max.z - entity->rect_extent.min.z) * 0.5f, 0.01f);
}


private void
collider_query_entity(
	collider_t collider,
	collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	if(collider->query_entities_used >= collider->query_entities_size)
	{
		uint32_t new_size = (collider->query_entities_size << 1) | 1;

		collider->query_entities = alloc_remalloc(collider->query_entities, collider->query_entities_size, new_size);
		assert_not_null(collider->query_entities);

		collider->query_entities_size = new_size;
	}

	collider->query_entities[collider->query_entities_used++] = entity;
}


private void
collider_query_fn(
	octree_t* ot,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(ot);
	assert_not_null(entity);

	(void) entity_idx;

	collider_t collider = MACRO_CONTAINER_OF(ot, struct collider, octree);
	collider_query_entity(collider, entity);
}


private void
collider_query(
	collider_t collider,
	rect_extent_3d_t extent
	)
{
	assert_not_null(collider);

	collider->query_entities_used = 0;

	octree_query(&collider->octree, extent, (void*) collider_query_fn);

	if(collider->query_entities_used < collider->query_entities_size / 4)
	{
		uint32_t new_size = collider->query_entities_size >> 1;

		collider->query_entities = alloc_remalloc(collider->query_entities, collider->query_entities_size, new_size);
		assert_not_null(collider->query_entities);

		collider->query_entities_size = new_size;
	}
}


private triplet_t
collider_closest_point_on_triangle(
	triplet_t p,
	triplet_t a,
	triplet_t b,
	triplet_t c
	)
{
	triplet_t ab = triplet_sub(b, a);
	triplet_t ac = triplet_sub(c, a);
	triplet_t ap = triplet_sub(p, a);

	float d1 = triplet_dot(ab, ap);
	float d2 = triplet_dot(ac, ap);
	if(d1 <= 0.0f && d2 <= 0.0f)
	{
		return a;
	}

	triplet_t bp = triplet_sub(p, b);

	float d3 = triplet_dot(ab, bp);
	float d4 = triplet_dot(ac, bp);
	if(d3 >= 0.0f && d4 <= d3)
	{
		return b;
	}

	triplet_t cp = triplet_sub(p, c);

	float d5 = triplet_dot(ab, cp);
	float d6 = triplet_dot(ac, cp);
	if(d6 >= 0.0f && d5 <= d6)
	{
		return c;
	}

	float vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float v = d1 / (d1 - d3);
		return triplet_add(a, triplet_scale(ab, v));
	}

	float vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float v = d2 / (d2 - d6);
		return triplet_add(a, triplet_scale(ac, v));
	}

	float va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
	{
		float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return triplet_add(b, triplet_scale(triplet_sub(c, b), v));
	}

	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;

	return triplet_add(a, triplet_add(triplet_scale(ab, v), triplet_scale(ac, w)));
}


private triplet_t
collider_get_surface_velocity(
	triplet_t center,
	triplet_t velocity,
	triplet_t angular_velocity,
	triplet_t contact_point
	)
{
	triplet_t radius_vec = triplet_sub(contact_point, center);
	triplet_t surface_vel = triplet_cross(angular_velocity, radius_vec);
	return triplet_add(velocity, surface_vel);
}


private void
collider_ball_collide_entity(
	collider_t collider,
	collider_entity_t* entity_a,
	collider_entity_t* entity_b
	)
{
	assert_not_null(entity_a);
	assert_not_null(entity_b);

	if(entity_a == entity_b)
	{
		return;
	}

	triplet_t center_a = rect_extent_3d_center(entity_a->rect_extent);


	switch(entity_b->type)
	{

	case  COLLIDER_ENTITY_TYPE_TRIANGLE:
	{
		triplet_t contact = collider_closest_point_on_triangle(
			center_a, entity_b->v0, entity_b->v1, entity_b->v2);

		triplet_t to_center = triplet_sub(center_a, contact);
		float distance = triplet_length(to_center);

		if(distance >= collider->ball_radius)
		{
			break;
		}

		entity_a->hit = true;

		triplet_t collision_normal = triplet_normalize(to_center);
		float penetration_depth = collider->ball_radius - distance;
		entity_a->pos_diff = triplet_add(entity_a->pos_diff, triplet_scale(collision_normal, penetration_depth));
		++entity_a->pos_diff_count;

		triplet_t relative_surface_vel = collider_get_surface_velocity(center_a, entity_a->v, entity_a->w, contact);
		float relative_normal_vel = triplet_dot(relative_surface_vel, collision_normal);

		if(relative_normal_vel >= 0.0f)
		{
			break;
		}

		float impulse_j = COLLISION_IMPULSE_FACTOR * relative_normal_vel;
		triplet_t normal_impulse_vec = triplet_scale(collision_normal, impulse_j);

		triplet_t tangential_vel = triplet_sub(relative_surface_vel, triplet_scale(collision_normal, relative_normal_vel));
		float tangential_speed = triplet_length(tangential_vel);
		triplet_t friction_impulse_vec = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

		if(tangential_speed > 0.000001f)
		{
			triplet_t tangent_dir = triplet_scale(tangential_vel, 1.0f / tangential_speed);
			float max_friction_j = COLLIDER_FRICTION * fabsf(impulse_j);
			float friction_j = fminf(max_friction_j, tangential_speed);
			friction_impulse_vec = triplet_scale(tangent_dir, -friction_j);
		}

		triplet_t total_impulse = triplet_add(normal_impulse_vec, friction_impulse_vec);
		entity_a->v_force = triplet_add(entity_a->v_force, total_impulse);

		triplet_t radius_vec = triplet_scale(collision_normal, -collider->ball_radius);
		triplet_t torque_impulse = triplet_cross(radius_vec, friction_impulse_vec);
		entity_a->w_force = triplet_add(entity_a->w_force, triplet_scale(torque_impulse, collider->inv_ball_moment_inertia));

		++entity_a->force_count;

		break;
	}

	case COLLIDER_ENTITY_TYPE_SPHERE:
	{
		triplet_t center_b = rect_extent_3d_center(entity_b->rect_extent);
		triplet_t penetration = triplet_sub(center_a, center_b);
		float distance = triplet_length(penetration);

		if(distance >= collider->ball_diameter)
		{
			break;
		}

		triplet_t normal = triplet_normalize(penetration);
		float penetration_depth = collider->ball_diameter - distance;

		float vel_a = fabsf(triplet_dot(entity_a->v, normal));
		float vel_b = fabsf(triplet_dot(entity_b->v, normal));
		float total_vel = vel_a + vel_b;

		float ratio_a = 0.5f;
		float ratio_b = 0.5f;
		if(total_vel > 0.0001f)
		{
			ratio_a = vel_b / total_vel;
			ratio_b = vel_a / total_vel;
		}

		triplet_t pos_diff_a = triplet_scale(normal, penetration_depth * ratio_a);
		triplet_t pos_diff_b = triplet_scale(normal, penetration_depth * ratio_b);
		entity_a->pos_diff = triplet_add(entity_a->pos_diff, pos_diff_a);
		++entity_a->pos_diff_count;
		entity_b->pos_diff = triplet_sub(entity_b->pos_diff, pos_diff_b);
		++entity_b->pos_diff_count;

		triplet_t r_a = triplet_scale(normal, -collider->ball_radius);
		triplet_t r_b = triplet_scale(normal, collider->ball_radius);
		triplet_t v_a_contact = triplet_add(entity_a->v, triplet_cross(entity_a->w, r_a));
		triplet_t v_b_contact = triplet_add(entity_b->v, triplet_cross(entity_b->w, r_b));
		triplet_t relative_surface_vel = triplet_sub(v_a_contact, v_b_contact);

		float relative_normal_vel = triplet_dot(relative_surface_vel, normal);
		if(relative_normal_vel >= 0.0f)
		{
			break;
		}

		float impulse_j = COLLISION_IMPULSE_FACTOR * relative_normal_vel;
		triplet_t normal_impulse_vec = triplet_scale(normal, impulse_j * 0.5f);

		triplet_t tangential_vel = triplet_sub(relative_surface_vel, triplet_scale(normal, relative_normal_vel));
		float tangential_speed = triplet_length(tangential_vel);
		triplet_t friction_impulse_vec = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

		if(tangential_speed > 0.000001f)
		{
			triplet_t tangent_dir = triplet_scale(tangential_vel, 1.0f / tangential_speed);
			float max_friction_j = COLLIDER_FRICTION * fabsf(impulse_j);
			float friction_j = fminf(max_friction_j, tangential_speed);
			friction_impulse_vec = triplet_scale(tangent_dir, -friction_j * 0.5f);
		}

		triplet_t total_impulse_a = triplet_add(normal_impulse_vec, friction_impulse_vec);
		entity_a->v_force = triplet_add(entity_a->v_force, total_impulse_a);
		entity_b->v_force = triplet_sub(entity_b->v_force, total_impulse_a);

		triplet_t torque_impulse_a = triplet_cross(r_a, friction_impulse_vec);
		triplet_t torque_impulse_b = triplet_cross(r_b, triplet_negate(friction_impulse_vec));

		entity_a->w_force = triplet_add(entity_a->w_force, triplet_scale(torque_impulse_a, collider->inv_ball_moment_inertia));
		entity_b->w_force = triplet_add(entity_b->w_force, triplet_scale(torque_impulse_b, collider->inv_ball_moment_inertia));

		++entity_a->force_count;
		++entity_b->force_count;

		break;
	}

	case COLLIDER_ENTITY_TYPE_FIST:
	{
		triplet_t center_b = rect_extent_3d_center(entity_b->rect_extent);
		triplet_t penetration = triplet_sub(center_a, center_b);
		float distance = triplet_length(penetration);
		float collision_distance = collider->ball_radius + collider->hand_radius;

		if(distance >= collision_distance)
		{
			break;
		}

		entity_a->hit = true;

		triplet_t collision_normal = triplet_normalize(penetration);
		float penetration_depth = collision_distance - distance;

		entity_a->pos_diff = triplet_add(entity_a->pos_diff, triplet_scale(collision_normal, penetration_depth));
		++entity_a->pos_diff_count;

		triplet_t contact_point = triplet_sub(center_a, triplet_scale(collision_normal, collider->ball_radius));
		triplet_t v_a_contact = collider_get_surface_velocity(center_a, entity_a->v, entity_a->w, contact_point);
		triplet_t v_b_contact = entity_b->v;
		triplet_t relative_surface_vel = triplet_sub(v_a_contact, v_b_contact);

		float relative_normal_vel = triplet_dot(relative_surface_vel, collision_normal);

		if(relative_normal_vel >= 0.0f)
		{
			break;
		}

		float impulse_j = COLLISION_IMPULSE_FACTOR * relative_normal_vel;
		triplet_t normal_impulse_vec = triplet_scale(collision_normal, impulse_j);

		triplet_t tangential_vel = triplet_sub(relative_surface_vel, triplet_scale(collision_normal, relative_normal_vel));
		float tangential_speed = triplet_length(tangential_vel);
		triplet_t friction_impulse_vec = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

		if(tangential_speed > 0.000001f)
		{
			triplet_t tangent_dir = triplet_scale(tangential_vel, 1.0f / tangential_speed);
			float max_friction_j = COLLIDER_FRICTION * fabsf(impulse_j);
			float friction_j = fminf(max_friction_j, tangential_speed);
			friction_impulse_vec = triplet_scale(tangent_dir, -friction_j);
		}

		triplet_t total_impulse = triplet_add(normal_impulse_vec, friction_impulse_vec);
		entity_a->v_force = triplet_add(entity_a->v_force, total_impulse);

		triplet_t radius_vec = triplet_scale(collision_normal, -collider->ball_radius);
		triplet_t torque_impulse = triplet_cross(radius_vec, friction_impulse_vec);
		entity_a->w_force = triplet_add(entity_a->w_force, triplet_scale(torque_impulse, collider->inv_ball_moment_inertia));

		++entity_a->force_count;

		break;
	}

	default: assert_unreachable();

	}
}


private bool
collider_ball_resolve(
	collider_t collider,
	collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	triplet_t pos = rect_extent_3d_center(entity->rect_extent);

	if(entity->pos_diff_count)
	{
		entity->pos_diff = triplet_scale(entity->pos_diff, 1.0f / entity->pos_diff_count);
		entity->pos_diff_count = 0;
	}

	pos = triplet_add(pos, entity->pos_diff);
	entity->pos_diff = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	if(entity->force_count)
	{
		float inv_force_count = 1.0f / entity->force_count;
		entity->v_force = triplet_scale(entity->v_force, inv_force_count);
		entity->w_force = triplet_scale(entity->w_force, inv_force_count);
		entity->force_count = 0;
	}

	entity->v = triplet_add(entity->v, entity->v_force);
	entity->w = triplet_add(entity->w, entity->w_force);
	entity->v_force = (triplet_t){{ 0.0f, 0.0f, 0.0f }};
	entity->w_force = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	if(entity->hit)
	{
		if(triplet_length(entity->w) > COLLIDER_LOW_ANGULAR_THRESHOLD)
		{
			triplet_t ideal_roll_axis = triplet_cross((triplet_t){{ 0.0f, 1.0f, 0.0f }}, entity->v);
			triplet_t ideal_w = triplet_scale(ideal_roll_axis, 1.0f / collider->ball_radius);

			entity->w = triplet_add(
				triplet_scale(entity->w, 1.0f - COLLIDER_ROLL_ALIGN_FACTOR),
				triplet_scale(ideal_w, COLLIDER_ROLL_ALIGN_FACTOR));
		}
	}
	else
	{
		triplet_t magnus_force = triplet_cross(entity->w, entity->v);
		entity->v = triplet_add(entity->v, triplet_scale(magnus_force, COLLIDER_MAGNUS_FACTOR));
	}

	triplet_t a = (triplet_t){{ 0.0f, -0.1f, 0.0f }};
	entity->v = triplet_add(entity->v, triplet_scale(a, collider->delta));
	entity->w = triplet_scale(entity->w, COLLIDER_ANGULAR_DAMPING);

	float speed = triplet_length(entity->v);
	if(speed > COLLIDER_VELOCITY_LIMIT)
	{
		entity->v = triplet_scale(entity->v, COLLIDER_VELOCITY_LIMIT / speed);
	}
	else if(
		speed < COLLIDER_LOW_SPEED_THRESHOLD &&
		triplet_length(entity->w) < COLLIDER_LOW_ANGULAR_THRESHOLD
		)
	{
		entity->v = triplet_scale(entity->v, COLLIDER_LOW_SPEED_DAMPING);
		entity->w = triplet_scale(entity->w, COLLIDER_LOW_SPEED_DAMPING);
	}

	pos = triplet_add(pos, triplet_scale(entity->v, collider->delta));

	if(pos.y < -1000.0f)
	{
		pos = (triplet_t){{ -1000.0f, 300.0f, -100.0f }};

		triplet_t v = {{ (float) rand() / RAND_MAX - 0.5f,
			(float) rand() / RAND_MAX - 0.5f,
			(float) rand() / RAND_MAX - 0.5f }};
		entity->v = triplet_scale(triplet_normalize(v), 10.0f);

		entity->w = (triplet_t){{ 0.0f, 0.0f, 0.0f }};
	}

	entity->rect_extent =
	(rect_extent_3d_t)
	{
		.min = {{ pos.x - collider->ball_radius, pos.y - collider->ball_radius, pos.z - collider->ball_radius }},
		.max = {{ pos.x + collider->ball_radius, pos.y + collider->ball_radius, pos.z + collider->ball_radius }}
	};

	if(entity->pos_external)
	{
		*entity->pos_external = pos;
	}

	if(entity->rotation_external)
	{
		triplet_t rotation = *entity->rotation_external;
		rotation = triplet_add(rotation, triplet_scale(entity->w, collider->delta));
		rotation.x = fmodf(rotation.x, M_PI * 2.0f);
		rotation.y = fmodf(rotation.y, M_PI * 2.0f);
		rotation.z = fmodf(rotation.z, M_PI * 2.0f);
		*entity->rotation_external = rotation;
	}

	return true;
}


private void
collider_collide(
	collider_t collider
	)
{
	assert_not_null(collider);

	collider_entity_t* ball = collider->balls;
	collider_entity_t* ball_end = ball + collider->balls_used;

	while(ball != ball_end)
	{
		ball->hit = false;

		collider_query(collider, ball->rect_extent);

		for(uint32_t i = 0; i < collider->query_entities_used; ++i)
		{
			collider_ball_collide_entity(collider, ball, collider->query_entities[i]);
		}

		collider_entity_t* other_ball = ball + 1;

		while(other_ball != ball_end)
		{
			collider_ball_collide_entity(collider, ball, other_ball);

			++other_ball;
		}

		++ball;
	}
}


private void
collider_resolve(
	collider_t collider
	)
{
	assert_not_null(collider);

	collider_entity_t* ball = collider->balls;
	collider_entity_t* ball_end = ball + collider->balls_used;

	while(ball != ball_end)
	{
		collider_ball_resolve(collider, ball);

		++ball;
	}
}


private void
collider_add_fist(
	collider_t collider
	)
{
	assert_not_null(collider);

	half_extent_3d_t extent =
	{
		.pos = collider->hand_pos,
		.size = {{ collider->hand_radius, collider->hand_radius, collider->hand_radius }}
	};

	collider_entity_t entity =
	{
		.type = COLLIDER_ENTITY_TYPE_FIST,

		.rect_extent = half_to_rect_3d_extent(extent),

		.v = collider->hand_v
	};

	collider_add(collider, &entity);
}


private void
collider_del_fist(
	collider_t collider
	)
{
	assert_not_null(collider);

	--collider->balls_used;
}


private void
collider_grab_ball(
	collider_t collider
	)
{
	assert_not_null(collider);

	if(!collider->grab)
	{
		return;
	}

	collider_entity_t* closest_ball = NULL;
	float min_dist_sq = FLT_MAX;

	collider_entity_t* ball = collider->balls;
	collider_entity_t* ball_end = ball + collider->balls_used;

	while(ball != ball_end)
	{
		triplet_t center = rect_extent_3d_center(ball->rect_extent);
		triplet_t to_hand = triplet_sub(collider->hand_pos, center);
		float dist_sq = triplet_dot(to_hand, to_hand);

		if(dist_sq < min_dist_sq)
		{
			min_dist_sq = dist_sq;
			closest_ball = ball;
		}

		++ball;
	}

	float radius = collider->hand_radius + collider->ball_radius;
	if(!closest_ball || min_dist_sq > radius * radius)
	{
		return;
	}

	closest_ball->v = collider->hand_v;
	closest_ball->w = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	closest_ball->rect_extent =
	(rect_extent_3d_t)
	{
		.min = {{ collider->hand_pos.x - collider->ball_radius,
			collider->hand_pos.y - collider->ball_radius,
			collider->hand_pos.z - collider->ball_radius }},
		.max = {{ collider->hand_pos.x + collider->ball_radius,
			collider->hand_pos.y + collider->ball_radius,
			collider->hand_pos.z + collider->ball_radius }}
	};

	if(closest_ball->pos_external)
	{
		*closest_ball->pos_external = collider->hand_pos;
	}

	if(closest_ball->rotation_external)
	{
		*closest_ball->rotation_external = (triplet_t){{ 0.0f, 0.0f, 0.0f }};
	}
}


void
collider_update(
	collider_t collider,
	float delta
	)
{
	collider->delta = delta;

	if(collider->fist)
	{
		collider_add_fist(collider);
	}

	uint64_t start = time_get();
	octree_normalize(&collider->octree);
	stats_log(collider->stats, "collider_normalize", time_get() - start);

	start = time_get();
	collider_collide(collider);
	stats_log(collider->stats, "collider_collide", time_get() - start);

	if(collider->fist)
	{
		collider_del_fist(collider);
	}

	start = time_get();
	collider_resolve(collider);
	stats_log(collider->stats, "collider_resolve", time_get() - start);

	if(collider->grab)
	{
		collider_grab_ball(collider);
	}
}


void
collider_set_hand(
	collider_t collider,
	triplet_t pos,
	bool fist,
	bool grab
	)
{
	assert_not_null(collider);

	collider->hand_vs[collider->hand_v_index] = triplet_scale(triplet_sub(pos, collider->hand_pos), COLLIDER_HAND_V_MULTIPLIER);
	collider->hand_v_index = (collider->hand_v_index + 1) % COLLIDER_HAND_V_COUNT;
	collider->hand_v_count = MACRO_MIN(collider->hand_v_count + 1, COLLIDER_HAND_V_COUNT);

	triplet_t avg_v = (triplet_t){{ 0.0f, 0.0f, 0.0f }};
	for(uint32_t i = 0; i < COLLIDER_HAND_V_COUNT; ++i)
	{
		avg_v = triplet_add(avg_v, collider->hand_vs[i]);
	}

	if(collider->hand_v_count > 0)
	{
		avg_v = triplet_scale(avg_v, 1.0f / collider->hand_v_count);
	}

	collider->hand_v = avg_v;

	collider->hand_pos = pos;
	collider->fist = fist;
	collider->grab = grab;
}
