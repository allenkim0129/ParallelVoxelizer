

#pragma once

#include "Octree.h"
#include <vector>

class OctreeBuilder
{
public:
	OctreeBuilder()
	~OctreeBuilder();
	Octree* buildOctree(const std::vector<glm::fmat3x3>& tri_meshes, 
		                float voxel_size);

private:
	//recursively subdivide pNode until it satisfy the base conditions:
	// 1. No interserction
	// 2. Node_size == voxel_size
	void  recSubDivide(const std::vector<glm::fmat3x3>& tri_meshes,
					   Node* pNode);

	
	glm::fmat3x3*	m_d_tri_meshes;
	uint 			m_num_tri_meshes; 
	Octree* 		m_octree;
}