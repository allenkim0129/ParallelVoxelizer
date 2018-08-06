
#include "OctreeBuilder.h"
#include "voxelizationMath.cu"

/** Forward declaration of CUDA calls **/
//Return AABB of triangle meshes
extern void getBoundingBox(uint n, const glm::fmat3x3* d_tri_meshes, 
	                       glm::fmat2x3& bounding_box); 

//Return true if there is a intersection between a node and triangle meshes
extern bool isBoxleIntersectsTriangle(uint n, const glm::fmat3x3* d_tri_meshes,
									  const glm::fvec3& min,const glm::fvec3& max);

//Return the shortest dist between a node and triangle meshes.
extern float calcLeafNodeSignedDistance(uint n, const glm::fmat3x3* d_tri_meshes,
									    const glm::fvec3& center);
////////////////////////////////////////////////////////////////////////////////////

Octree* OctreeBuilder::buildOctree(const std::vector<glm::fmat3x3>& tri_meshes, 
		                           float voxel_size)
{
	//copy tri_meshes to divice pointer
	m_num_tri_meshes = tri_meshes.size();
	cudaMalloc(&m_d_tri_meshes, m_num_tri_meshes*sizeof(glm::fmat3x3));
	cudaMemcpy(m_d_tri_meshes, tri_meshes.data(), m_num_tri_meshes*sizeof(glm::fmat3x3));

	//create a new octree
	m_octree = new Octree(voxel_size);

	//get AABB of triangle meshes to create a root node
	Node* root = m_octree.getRoot();
	root = new Node();
	glm::fmat2x3 bounding_box;
	getBoundingBox(m_num_tri_meshes, m_d_tri_meshes, bounding_box);
	root->AABB = bounding_box;

	//start subdivide
	recSubDivision(root);
}

void OctreeBuilder::recSubDivision(Node* pNode)
{
	//parallel
	bool is_intersect = isBoxleIntersectsTriangle(m_num_tri_meshes, d_tri_meshes, 
												  pNode->m_box.min,
												  pNode->m_box.max);

	if (is_intersect)
	{
		//LEAF Nodes
		if (pNode->dim == octree->voxel_dim())		// default voxel_dim is '1'
		{
			pNode->m_node_type = NT_LEAF;
			//parallel
			float dist = calcLeafNodeSignedDistance(m_num_tri_meshes, d_tri_meshes,
				                                    pNode->m_box.center);
			pNode->m_signed_dist = dist;
		}
		//Internel Nodes
		else
		{
			pNode->m_node_type   = NT_INTERNEL;
			pNode->m_signed_dist = 0.0f;
			pNode->m_children    = new Node[8];

			for (int i = 0; i < 8; ++i)
			{
				Node* cNode = new Node();
				cNode.m_parent = pNode;
				setAABB(cNode, i);			//set m_box for current child Node
				pNode->m_children[i] = cNode;
				recSubDivision(cNode);
			}
		}
	}
	//EMPTY Nodes
	else
	{
		pNode->m_type = NT_EMPTY;
		//parallel
		float dist = calcLeafNodeSignedDistance(m_num_tri_meshes, d_tri_meshes,
			                                    pNode->m_box.center);
		pNode->m_signed_dist = dist;
	}
}