
__global__ void SFEGO_3d_gradient(float *data, float *diff,
						float *direct_x, float *direct_y, float *direct_z,
						int *list_x, int *list_y, int *list_z,
						float *list_unit_x, float *list_unit_y, float *list_unit_z,
						int *surface_indexs,
						int *hemisphere_dp_pos_add, int *hemisphere_dp_pos_sub,
						int *hemisphere_dp_neg_add, int *hemisphere_dp_neg_sub,
						int *dp_pos_add_start_idxs, int *dp_pos_sub_start_idxs,
						int *dp_neg_add_start_idxs, int *dp_neg_sub_start_idxs,
						int *dp_pos_add_start_lens, int *dp_pos_sub_start_lens, 
						int *dp_neg_add_start_lens, int *dp_neg_sub_start_lens,
						int list_len, int dp_len, int dim_x, int dim_y, int dim_z
						) {
	//kernel index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;

	//np_data[z*dim_y*dim_x+y*dim_x+x]

	if(x<dim_x && y<dim_y && z<dim_z){
		//init variable
		float pos_sum=0.0, neg_sum=0.0;
		int pos_count=0, neg_count=0;
		float avg_pos, avg_neg, avg_diff;
		float max_diff=-99999.9;
		int max_index;
		float current=data[z*dim_y*dim_x+y*dim_x+x];
		for(int i=0;i<dp_len;++i){
			for(int j=0;j<dp_pos_add_start_lens[i];++j){
				int idx=j+dp_pos_add_start_idxs[i];
				int index=hemisphere_dp_pos_add[idx];
				int tx=x+list_x[index];
				int ty=y+list_y[index];
				int tz=z+list_z[index];
				if( (tx>=0 && tx<dim_x) && (ty>=0 && ty<dim_y) && (tz>=0 && tz<dim_z)){
					pos_sum+=data[tz*dim_y*dim_x+ty*dim_x+tx];
					pos_count++;
				}
			}
			for(int j=0;j<dp_pos_sub_start_lens[i];++j){
				int idx=j+dp_pos_sub_start_idxs[i];
				int index=hemisphere_dp_pos_sub[idx];
				int tx=x+list_x[index];
				int ty=y+list_y[index];
				int tz=z+list_z[index];
				if( (tx>=0 && tx<dim_x) && (ty>=0 && ty<dim_y) && (tz>=0 && tz<dim_z)){
					pos_sum-=data[tz*dim_y*dim_x+ty*dim_x+tx];
					pos_count--;
				}
			}
			for(int j=0;j<dp_neg_add_start_lens[i];++j){
				int idx=j+dp_neg_add_start_idxs[i];
				int index=hemisphere_dp_neg_add[idx];
				int tx=x+list_x[index];
				int ty=y+list_y[index];
				int tz=z+list_z[index];
				if( (tx>=0 && tx<dim_x) && (ty>=0 && ty<dim_y) && (tz>=0 && tz<dim_z)){
					neg_sum+=data[tz*dim_y*dim_x+ty*dim_x+tx];
					neg_count++;
				}
			}
			for(int j=0;j<dp_neg_sub_start_lens[i];++j){
				int idx=j+dp_neg_sub_start_idxs[i];
				int index=hemisphere_dp_neg_sub[idx];
				int tx=x+list_x[index];
				int ty=y+list_y[index];
				int tz=z+list_z[index];
				if( (tx>=0 && tx<dim_x) && (ty>=0 && ty<dim_y) && (tz>=0 && tz<dim_z)){
					neg_sum-=data[tz*dim_y*dim_x+ty*dim_x+tx];
					neg_count--;
				}
			}
			if(pos_count>0) avg_pos=pos_sum/pos_count;
			else avg_pos=current;
			if(neg_count>0) avg_neg=neg_sum/neg_count;
			else avg_neg=current;
			avg_diff=avg_pos-avg_neg;
			if(avg_diff>max_diff){
				max_diff=avg_diff;
				max_index=surface_indexs[i];
			}
		}
		//printf("%d %d %d => %d %f\n", x, y, z, max_index, max_diff);
		direct_x[z*dim_y*dim_x+y*dim_x+x]=list_unit_x[max_index];
		direct_y[z*dim_y*dim_x+y*dim_x+x]=list_unit_y[max_index];
		direct_z[z*dim_y*dim_x+y*dim_x+x]=list_unit_z[max_index];
		diff[z*dim_y*dim_x+y*dim_x+x]=max_diff;
	}
}


__global__ void SFEGO_3d_integral(float *result, float *diff,
						float *direct_x, float *direct_y, float *direct_z,
                        int *list_x, int *list_y, float *list_z,
						float *list_unit_x, float *list_unit_y, float *list_unit_z,
						int list_len, int dim_x, int dim_y, int dim_z) {
	
	//kernel index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	
	if(x<dim_x && y<dim_y && z<dim_z){
		int tx,ty,tz;
		result[z*dim_y*dim_x+y*dim_x+x]=0.0;
		for(int i=0;i<list_len;++i){
			tx=x+list_x[i];
			ty=y+list_y[i];
			tz=z+list_z[i];
			if( (tx>=0 && tx<dim_x) && (ty>=0 && ty<dim_y) && (tz>=0 && tz<dim_z) ){
				float dot_product=	(list_unit_x[i]*direct_x[tz*dim_y*dim_x+ty*dim_x+tx])+
									(list_unit_y[i]*direct_y[tz*dim_y*dim_x+ty*dim_x+tx])+
									(list_unit_z[i]*direct_z[tz*dim_y*dim_x+ty*dim_x+tx]);
				result[z*dim_y*dim_x+y*dim_x+x]+=dot_product*diff[tz*dim_y*dim_x+ty*dim_x+tx];
			}
		}
	}
}

