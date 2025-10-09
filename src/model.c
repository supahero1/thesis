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
#include <thesis/model.h>
#include <thesis/alloc_ext.h>

#include <assimp/scene.h>
#include <assimp/cimport.h>
#include <assimp/postprocess.h>


static const char*
const_basename(
	const char *path
	)
{
	const char* rslash = strrchr(path, '/');
	const char* lslash = strrchr(path, '\\');
	return rslash ? rslash + 1 : lslash ? lslash + 1 : path;
}


model_t*
model_init(
	const char* path,
	float scale
	)
{
	model_t* model = alloc_malloc(model, 1);
	assert_not_null(model);

	const struct aiScene* scene = aiImportFile(
		path,
		aiProcess_GenNormals |
		aiProcess_SortByPType |
		aiProcess_GenUVCoords |
		aiProcess_Triangulate |
		aiProcess_OptimizeGraph |
		aiProcess_OptimizeMeshes |
		aiProcess_FindDegenerates |
		aiProcess_FindInvalidData |
		aiProcess_TransformUVCoords |
		aiProcess_FixInfacingNormals |
		aiProcess_PreTransformVertices |
		aiProcess_ImproveCacheLocality |
		aiProcess_JoinIdenticalVertices |
		aiProcess_ValidateDataStructure |
		aiProcess_RemoveRedundantMaterials
	);
	hard_assert_not_null(scene, fprintf(stderr, "Failed to load model: %s\n", aiGetErrorString()));

	assert_not_null(scene->mRootNode);
	assert_eq(scene->mNumCameras, 0);
	assert_eq(scene->mNumLights, 0);
	assert_eq(scene->mNumAnimations, 0);
	assert_gt(scene->mNumMaterials, 0);
	assert_gt(scene->mNumMeshes, 0);

	printf("Model '%s':\n", path);

	model->material_count = scene->mNumMaterials;
	model->materials = alloc_malloc(model->materials, model->material_count);
	assert_ptr(model->materials, model->material_count);

	printf("- material_count: %u\n", model->material_count);
	printf("- materials:\n");

	for(uint32_t i = 0; i < model->material_count; i++)
	{
		material_t* material = &model->materials[i];

		printf("    - material[%u]:\n", i);

		const struct aiMaterial* sceneMaterial = scene->mMaterials[i];
		assert_not_null(sceneMaterial);

		struct aiColor4D color;

		enum aiReturn status = aiGetMaterialColor(sceneMaterial, AI_MATKEY_COLOR_DIFFUSE, &color);
		assert_eq(status, AI_SUCCESS);
		glm_vec3_copy((void*) &color, material->diffuse);

		printf("        - diffuse: (%.3f, %.3f, %.3f)\n",
			material->diffuse[0], material->diffuse[1], material->diffuse[2]);

		status = aiGetMaterialColor(sceneMaterial, AI_MATKEY_COLOR_AMBIENT, &color);
		if(status != AI_SUCCESS)
		{
			color.r = 0.0f;
			color.g = 0.0f;
			color.b = 0.0f;
		}
		glm_vec3_copy((void*) &color, material->ambient);

		printf("        - ambient: (%.3f, %.3f, %.3f)\n",
			material->ambient[0], material->ambient[1], material->ambient[2]);

		float shininess;
		status = aiGetMaterialFloatArray(sceneMaterial, AI_MATKEY_SHININESS, &shininess, NULL);
		if(status != AI_SUCCESS)
		{
			shininess = 32.0f;
		}
		material->shininess = shininess;

		printf("        - shininess: %.3f\n", material->shininess);

		float shininess_strength;
		status = aiGetMaterialFloatArray(sceneMaterial, AI_MATKEY_SHININESS_STRENGTH, &shininess_strength, NULL);
		if(status != AI_SUCCESS)
		{
			shininess_strength = 1.0f;
		}
		material->shininess_strength = shininess_strength;

		printf("        - shininess_strength: %.3f\n", material->shininess_strength);

		material->texture = str_init();

		struct aiString texture_path;
		if(AI_SUCCESS == aiGetMaterialTexture(sceneMaterial,
			aiTextureType_DIFFUSE, 0, &texture_path, NULL, NULL, NULL, NULL, NULL, NULL))
		{
			const char* path_dir_end = const_basename(path) - 1;
			size_t path_len = path_dir_end - path;

			const char* tex_path = const_basename(texture_path.data);
			size_t tex_len = texture_path.length - (tex_path - texture_path.data);

			char* combined_path = alloc_malloc(combined_path, path_len + tex_len + 2);
			assert_not_null(combined_path);

			char* ptr = combined_path;
			memcpy(ptr, path, path_len);
			ptr += path_len;
			*(ptr++) = '/';
			memcpy(ptr, tex_path, tex_len);
			ptr += tex_len;
			*(ptr++) = 0;

			str_set_move_len(material->texture, combined_path, ptr - combined_path);
		}

		printf("        - texture: '%s'\n", (char*) material->texture->str);
	}

	model->mesh_count = scene->mNumMeshes;
	model->meshes = alloc_malloc(model->meshes, model->mesh_count);
	assert_ptr(model->meshes, model->mesh_count);

	printf("- mesh_count: %u\n", model->mesh_count);
	printf("- meshes:\n");

	for(uint32_t i = 0; i < model->mesh_count; i++)
	{
		mesh_t* mesh = &model->meshes[i];

		printf("    - mesh[%u]:\n", i);

		const struct aiMesh* sceneMesh = scene->mMeshes[i];
		assert_not_null(sceneMesh);

		assert_eq(sceneMesh->mPrimitiveTypes, aiPrimitiveType_TRIANGLE);
		assert_not_null(sceneMesh->mTextureCoords[0]);

		mesh->material_idx = sceneMesh->mMaterialIndex;
		mesh->vertex_count = sceneMesh->mNumVertices;
		assert_gt(mesh->vertex_count, 0);

		printf("        - material_idx: %u\n", mesh->material_idx);
		printf("        - material->texture: '%s'\n",
			(char*) model->materials[mesh->material_idx].texture->str);
		printf("        - vertex_count: %u\n", mesh->vertex_count);

		mesh->vertices = alloc_malloc(mesh->vertices, mesh->vertex_count);
		assert_ptr(mesh->vertices, mesh->vertex_count);

		mesh->normals = alloc_malloc(mesh->normals, mesh->vertex_count);
		assert_ptr(mesh->normals, mesh->vertex_count);

		mesh->coords = alloc_malloc(mesh->coords, mesh->vertex_count);
		assert_ptr(mesh->coords, mesh->vertex_count);

		for(uint32_t j = 0; j < mesh->vertex_count; j++)
		{
			glm_vec3_copy((void*) &sceneMesh->mVertices[j], mesh->vertices[j]);
			glm_vec3_copy((void*) &sceneMesh->mNormals[j], mesh->normals[j]);
			glm_vec2_copy((void*) &sceneMesh->mTextureCoords[0][j], mesh->coords[j]);

			mesh->vertices[j][0] *= scale;
			mesh->vertices[j][1] *= scale;
			mesh->vertices[j][2] *= scale;
		}

		mesh->index_count = sceneMesh->mNumFaces * 3;
		mesh->indexes = alloc_malloc(mesh->indexes, mesh->index_count);
		assert_ptr(mesh->indexes, mesh->index_count);

		printf("        - index_count: %u\n", mesh->index_count);

		for(uint32_t j = 0; j < sceneMesh->mNumFaces; j++)
		{
			const struct aiFace* face = &sceneMesh->mFaces[j];
			assert_eq(face->mNumIndices, 3);

			mesh->indexes[j * 3 + 0] = face->mIndices[0];
			mesh->indexes[j * 3 + 1] = face->mIndices[1];
			mesh->indexes[j * 3 + 2] = face->mIndices[2];
		}
	}

	aiReleaseImport(scene);

	puts("");

	return model;
}


void
model_free(
	model_t* model
	)
{
	assert_not_null(model);

	for(uint32_t i = 0; i < model->mesh_count; i++)
	{
		mesh_t* mesh = &model->meshes[i];
		assert_not_null(mesh);

		alloc_free(mesh->indexes, mesh->index_count);

		alloc_free(mesh->coords, mesh->vertex_count);
		alloc_free(mesh->normals, mesh->vertex_count);
		alloc_free(mesh->vertices, mesh->vertex_count);
	}

	alloc_free(model->meshes, model->mesh_count);

	for(uint32_t i = 0; i < model->material_count; i++)
	{
		material_t* material = &model->materials[i];
		assert_not_null(material);

		str_free(material->texture);
	}

	alloc_free(model->materials, model->material_count);

	alloc_free(model, 1);
}
