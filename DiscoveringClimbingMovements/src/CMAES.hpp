#include <Eigen/Eigen>
#include "ProbUtils.hpp"
#include "MiscUtils.hpp"


#include <algorithm>
#include <iterator>
#include <math.h>


#ifndef CMAES_H
#define CMAES_H

enum Operation_mode{
	BEST, MEAN, EXTERNAL
};

template<typename Scalar>
class CMAES{

	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
	typedef std::pair<Vector, Scalar> ScoredData; //The scores are used only in sorting and do not have signifigance as weights.

private:

	Vector evolution_path_;
	Vector evolution_path_conjugate_;

	static bool hasLowerScore(const ScoredData& datum1, const ScoredData& datum2){
		return datum1.second < datum2.second;
	}

	static bool hasHigherScore(const ScoredData& datum1, const ScoredData& datum2){
		return datum1.second > datum2.second;
	}


public:

	Matrix covariance_;
	Vector mean_;
	Scalar selected_samples_fraction_;
	bool initialized_;
	bool diagonal_;
	bool decomposition_computed_;
	std::pair<Vector,Matrix> sampling_decomposition_;
	Vector maximum_standard_deviations_;
	Scalar weight_scaling_;

	Scalar step_size_;
	bool use_step_size_control_;
	Scalar mu_eff;
	Scalar cc_;
	Scalar cs_;
	Scalar c1_;
	Scalar cmu_;
	Scalar damping_;
	Scalar chiN_;
	Operation_mode operation_mode_;

	Scalar min_cmu_;
	Scalar max_cmu_;
	Scalar max_step_size_;
	Scalar min_step_size_;
	Scalar minimum_exploration_variance_;
	Scalar weight_decay_;

	//Scalar forgetting_factor_complementary_evolution_path_;
	//Scalar damping_parameter_;

	std::vector<Scalar> weights_;

	CMAES(unsigned dimension = 0){
		init_with_dimension(dimension);
	}

	void init_with_dimension(unsigned dimension){
		mean_ = Vector::Zero(dimension);
		evolution_path_ = mean_;
		covariance_ = Matrix::Zero(dimension,dimension);
		cc_ = (Scalar)0.3333;
		cs_ = (Scalar)0.3333;
		c1_ = (Scalar)0.3333;
		cmu_ = (Scalar)0.3333;
		min_cmu_ = (Scalar)0.75;
		max_cmu_ = (Scalar)1.0;
		//damping_parameter_ = (Scalar)1.0;
		diagonal_ = false;
		step_size_ = (Scalar)1.0;
		initialized_ = false;
		decomposition_computed_ = false;
		sampling_decomposition_ = std::make_pair(mean_,covariance_);
		maximum_standard_deviations_ = Vector::Ones(dimension)*std::numeric_limits<Scalar>::infinity();
		weight_scaling_ = (Scalar)3.0;
		mu_eff = 1;
		damping_ = 1;
		chiN_ = unbiasedSelectionExpectation((float)dimension);
		evolution_path_ = Vector::Zero(mean_.size());
		evolution_path_conjugate_ = Vector::Zero(mean_.size());
		use_step_size_control_ = false;
		operation_mode_ = Operation_mode::MEAN;
		max_step_size_ = (Scalar)10;
		min_step_size_ = (Scalar)0.1;
		minimum_exploration_variance_ = (Scalar)0.01;
		minimum_exploration_variance_ = minimum_exploration_variance_*minimum_exploration_variance_;
		weight_decay_ = (Scalar)0.5;

	}

	Scalar unbiasedSelectionExpectation(Scalar dimension){
		Scalar result = std::sqrt(dimension) * ( (Scalar)1 - (Scalar)1 / ( (Scalar)4*dimension) + (Scalar)1 / ( (Scalar)21 * dimension*dimension) );
		return result;
	}

	Scalar kurtosis(const std::vector<ScoredData*>& data_with_score, int data_amount){
		Scalar weight_mean = (Scalar)0.0;
		for (int i = 0; i < data_amount; i++){
			weight_mean += data_with_score[i]->second;
		}
		weight_mean /= (Scalar)data_amount;

		Scalar weight_var = (Scalar)0;
		for (int i = 0; i < data_amount; i++){
			weight_var += std::pow(data_with_score[i]->second - weight_mean,(Scalar)2);
		}
		weight_var = weight_var/(Scalar)data_amount;

		Scalar weight_fourth_moment = (Scalar)0;
		for (int i = 0; i < data_amount; i++){
			weight_fourth_moment += std::pow(data_with_score[i]->second - weight_mean,(Scalar)4);
		}
		weight_fourth_moment = weight_fourth_moment/(Scalar)data_amount;

		Scalar weight_kurtosis = weight_fourth_moment / (weight_var*weight_var);
		return weight_kurtosis;
	}

	void setMean(const Vector& new_mean){
		mean_ = new_mean;
		evolution_path_ = Vector::Zero(mean_.size());
		evolution_path_conjugate_ = Vector::Zero(mean_.size());
	}

	void setMaximumDeviations(const Vector& maximums){
		maximum_standard_deviations_ = maximums.cwiseAbs();
	}

	void update(std::vector<ScoredData >& data_with_score, bool use_weights = true, const Vector& extern_mean = Vector::Zero(0)){

		decomposition_computed_ = false;

		std::vector<ScoredData*> data_with_score_ptr;
		data_with_score_ptr.reserve(data_with_score.size());
		for (ScoredData& datum : data_with_score){
			data_with_score_ptr.push_back(&datum);
		}

		//auto has_lower_score_ptr = [&](const ScoredData* ptr_1, const ScoredData* ptr_2){

		//	if (!ptr_1 && !ptr_2){
		//		return false;
		//	}
		//	else if (ptr_1 && !ptr_2){
		//		return true;
		//	}
		//	else if (!ptr_1 && ptr_2){
		//		return false;
		//	}
		//	else{
		//		if (ptr_1->second < ptr_2->second){
		//			return true;
		//		}
		//		else{
		//			return false;
		//		}
		//	}

		//};

		auto has_higher_score_ptr = [&](const ScoredData* ptr_1, const ScoredData* ptr_2){
			return ptr_1->second > ptr_2->second;
			//if (!ptr_1 && !ptr_2){
			//	return false;
			//}
			//else if (ptr_1 && !ptr_2){
			//	return true;
			//}
			//else if (!ptr_1 && ptr_2){
			//	return false;
			//}
			//else{
			//	if (ptr_1->second < ptr_2->second){
			//		return true;
			//	}
			//	else{
			//		return false;
			//	}
			//} 

		};
		auto it_begin = data_with_score_ptr.begin();
		auto it_end = data_with_score_ptr.end();

		std::sort(it_begin,it_end,has_higher_score_ptr);


		unsigned last_non_inf = 0;

		for (unsigned i = 0; i < data_with_score_ptr.size(); i++){
			if (finiteNumber( data_with_score_ptr[i]->second)){
				last_non_inf = i;
			}
			else{
				break;
			}
		}

		unsigned data_amount = std::max((unsigned)1, (unsigned)(selected_samples_fraction_*(Scalar)data_with_score_ptr.size()) );
		data_amount = std::min(data_amount,last_non_inf + 1);  


		Scalar weight_scaling = kurtosis(data_with_score_ptr,data_amount);
		weight_scaling = std::max(weight_scaling,(Scalar)weight_scaling_);
		weight_scaling = std::min(weight_scaling,std::numeric_limits<Scalar>::max());

		if (!finiteNumber(weight_scaling)){
			weight_scaling = weight_scaling_;
		}
		//weight_scaling = weight_scaling_; //NB! Debug only!!! //This wrked actually quite well in SaDDP!


		//Change the scores to probabilities using softmax
		weights_.clear();
		weights_.reserve(data_amount);
		if (use_weights){


			for (unsigned i = 0; i < data_amount; i++){
				weights_.push_back(data_with_score_ptr[i]->second);
			}
			assert(check_valid_vector(weights_));

			weights_ = softmax_scaled<Scalar>(weights_,-weight_scaling);


			//Scalar current_factor = 1;
			//for (unsigned i = 0; i < data_amount; i++){
			//	weights_.push_back(current_factor);
			//	current_factor *= weight_decay_;
			//}
			//Scalar weight_sum = sum(weights_);
			//for (unsigned i = 0; i < data_amount; i++){
			//	weights_[i] /= weight_sum;
			//}
			assert(check_valid_vector(weights_));
		}
		else{
			Scalar mu=(Scalar)data_amount; //"mu" in CMA-ES https://en.wikipedia.org/wiki/CMA-ES
			Scalar total=0;
			for (unsigned i = 0; i < data_amount; i++){
				Scalar w=log(mu+0.5f);
				w-=log((Scalar)(i+1));
				total+=w;
			}
			for (unsigned i = 0; i < data_amount; i++){
				Scalar w=log(mu+0.5f)-log((Scalar)(i+1));
				weights_.push_back(w/total);
			}

			assert(check_valid_vector(weights_));
		}

		mu_eff = (Scalar)1/squareSum(weights_);
		cc_ = defaultCC();
		cs_ = defaultCS();
		c1_ = defaultC1();
		cmu_ = defaultCMU();

		cmu_ = std::max(std::min(cmu_,max_cmu_),min_cmu_);
		c1_ = std::max((Scalar)0,std::min((Scalar)1,c1_));
		if (c1_ + cmu_ > 1){
			c1_ = 0;
			cmu_ = cmu_;
		}
		cc_ = std::max((Scalar)0,std::min((Scalar)1,cc_));
		cs_ = std::max((Scalar)0,std::min((Scalar)1,cs_));

		damping_ = defaultDampingFactor();

		assert(check_valid_vector(weights_));

		Vector next_gen_mean;
		//Compute new mean
		if (operation_mode_ == BEST){
			next_gen_mean = data_with_score_ptr[0]->first;
		}

		if (operation_mode_ == MEAN){
			next_gen_mean = meanUpdate(data_with_score_ptr,data_amount);
		}

		if (extern_mean.size() > 0){
			operation_mode_ = Operation_mode::EXTERNAL;
			next_gen_mean = extern_mean;
		}

		assert(check_valid_matrix(next_gen_mean));

		if (!initialized_){
			setMean(next_gen_mean);
			shiftData(data_with_score_ptr, mean_, (Scalar)1 );
			covariance_ = rankMuUpdate(data_with_score_ptr,data_amount);
			assert(check_valid_matrix(covariance_));
			if (diagonal_){
				Vector tmp = covariance_.diagonal();
				covariance_ = tmp.asDiagonal();
			}
			initialized_ = true;
			return;
		}

		//Center data
		shiftData(data_with_score_ptr, mean_, (Scalar)1/(step_size_) );
		//shiftData(data_with_score_ptr, next_gen_mean, (Scalar)1/(step_size_) );

		//Update evolution path
		computeEvolutionPath(next_gen_mean);
		if (use_step_size_control_){
			computeComplementaryEvolutionPath(next_gen_mean);
			computeStepSize();
		}

		mean_ = next_gen_mean;

		//Update covariance matrix
		covariance_ *= std::max(((Scalar)1 - cmu_ - c1_),(Scalar)0);
		covariance_ += std::max(c1_,(Scalar)0)*rankOneUpdate();
		covariance_ += std::max(cmu_,(Scalar)0)*rankMuUpdate(data_with_score_ptr,data_amount);

		assert(check_valid_matrix(covariance_));
		if (diagonal_){
			Vector tmp = covariance_.diagonal();
			covariance_ = tmp.asDiagonal();
		}
		assert(check_valid_matrix(covariance_));

	}

	void computeEvolutionPath(Vector& next_gen_mean){
		evolution_path_ *= ((Scalar)1 - cc_);

		sampling_decomposition_ = decompositionForSampling();
		decomposition_computed_ = false;

		assert(check_valid_matrix(evolution_path_));
		evolution_path_ += std::sqrt(cc_*((Scalar)2-cc_)*mu_eff) * (next_gen_mean - mean_) / step_size_;
		assert(check_valid_matrix(evolution_path_));
	}

	void computeComplementaryEvolutionPath(Vector& next_gen_mean){
		evolution_path_conjugate_ *= ((Scalar)1 - cs_);
		assert(check_valid_matrix(evolution_path_conjugate_));

		Matrix cov = covariance_;
		cov += Matrix::Identity(cov.rows(),cov.cols())*minimum_exploration_variance_;
		Matrix cov_inv = cov.inverse();

		Matrix cov_inv_sqrt = cov_inv.llt().matrixL();


		assert((Scalar)cs_ <= 1.0 && cs_ >= (Scalar)0);
		evolution_path_conjugate_ += std::sqrt(cs_*((Scalar)2-cs_)*mu_eff) * cov_inv_sqrt * (next_gen_mean - mean_) / step_size_;

		//If something goes wrong, just set the complementary evolution path to zero
		if (!check_valid_matrix(evolution_path_conjugate_)){
			evolution_path_conjugate_ = Vector::Zero(evolution_path_conjugate_.size());
		}
		assert(check_valid_matrix(evolution_path_conjugate_));
	}

	void shiftData(std::vector<ScoredData*>& data_with_score, const Vector& shift, const Scalar& scale){
		for (ScoredData* datum : data_with_score){
			datum->first -= shift;
			datum->first *= scale;
			assert(check_valid_matrix(datum->first));
		}
	}

	Vector meanUpdate(const std::vector<ScoredData*>& data_with_score, const unsigned& data_amount, bool use_weights = true){
		Vector new_mean = Vector::Zero(mean_.size());

		unsigned dataToConsider = std::min(data_amount,data_with_score.size());
		for (unsigned i = 0; i < dataToConsider; i++){
			const ScoredData& datum = *(data_with_score[i]);

			if (use_weights){
				new_mean += datum.first*weights_[i];
			}
			else{
				new_mean += datum.first;
			}

			assert(check_valid_matrix(new_mean));
		}

		if (!use_weights){
			new_mean /= (Scalar)dataToConsider;
		}

		return new_mean;

	}

	Matrix rankMuUpdate(const std::vector<ScoredData*>& centered_data_with_score, const unsigned& data_amount){

		Matrix update_matrix = Matrix::Zero(mean_.size(),mean_.size());

		unsigned dataToConsider = std::min(data_amount,centered_data_with_score.size());
		//#pragma omp parallel for
		for (unsigned i = 0; i < dataToConsider; i++){
			const ScoredData& datum = *(centered_data_with_score[i]);


			update_matrix += weights_[i] * datum.first * datum.first.transpose();


			assert(check_valid_matrix(update_matrix));
		}


		//Normalize the covariance
		Scalar weightSum = 0;
		Scalar weightSqSum = 0;
		for (Scalar& element : weights_){
			weightSqSum += element*element;
			weightSum += element;
		}

		//If all of the probability mass is concentrated to one sample, return slightly Tikhonov regularized zero matrix.
		if ((Scalar)1/weightSqSum < (Scalar)1+std::numeric_limits<Scalar>::epsilon()){
			update_matrix = (Vector::Zero(mean_.size())).asDiagonal();
			assert(check_valid_matrix(update_matrix));
			return update_matrix;
		}

		Scalar normalizationDenominator = weightSum*weightSum - weightSqSum;
		if (normalizationDenominator < std::numeric_limits<Scalar>::epsilon()){
			normalizationDenominator = std::numeric_limits<Scalar>::epsilon();
		}
		update_matrix *= ( weightSum / normalizationDenominator );
		assert(check_valid_matrix(update_matrix));


		return update_matrix;
	}

	Matrix rankOneUpdate(){
		assert(check_valid_matrix(evolution_path_));
		return evolution_path_ * evolution_path_.transpose();
	}

	Scalar getForgettingFactor(const Scalar& forgetting_factor){
		return std::max( std::min( forgetting_factor, (Scalar)1 ), (Scalar)0); //Clamp between 1 and 0;
	}

	void computeStepSize(){
		Scalar step_size_tmp = std::sqrt(evolution_path_conjugate_.dot(evolution_path_conjugate_)) / chiN_ - (Scalar)1;
		assert(finiteNumber(step_size_tmp));
		step_size_ *= std::exp(cs_ / damping_ * step_size_tmp);
		if (!finiteNumber(step_size_)){
			step_size_ = (Scalar)1;
		}
		assert(finiteNumber(step_size_));
		step_size_ = std::min(step_size_,max_step_size_);
		step_size_ = std::max(step_size_,min_step_size_);
	}

	Scalar defaultDampingFactor(void){
		Scalar option = std::sqrt( (mu_eff-(Scalar)1) / ( (Scalar)(mean_.size() + 1) ) ) - (Scalar)1;
		if (option - option != option - option){
			return (Scalar)1 + cs_;
		}

		return (Scalar)1 + cs_ + (Scalar)2 * std::max((Scalar)0,option);
	}

	Scalar defaultCS(){
		return (mu_eff + (Scalar)2) / ((Scalar)mean_.size() + mu_eff + (Scalar)5);
	}

	Scalar defaultCC(){
		Scalar weightToSizeRatio = mu_eff/(Scalar)mean_.size();
		return (4.0f + weightToSizeRatio) / ((Scalar)(mean_.size() + 4) + (Scalar)2.0*weightToSizeRatio);
	}

	Scalar defaultC1(){
		return (Scalar)2 / (std::pow((Scalar)mean_.size() + (Scalar)1.3, (Scalar)2 ) + mu_eff);
	}

	Scalar defaultCMU(){
		Scalar option = (Scalar)2 * (mu_eff - (Scalar)2 + (Scalar)1/mu_eff);
		option /= std::pow((Scalar)mean_.size() + (Scalar)2, (Scalar)2 ) + mu_eff;

		return std::min((Scalar)1 - c1_, option);
	}


	//Return the mean and the Cholesky decomposition of the covariance matrix so that we can sample from non-standard normal distribution.
	std::pair<Vector,Matrix> decompositionForSampling(){

		//assert(check_valid_matrix(covariance_));
		//scaleCovarianceMatrixToMaximum();
		//assert(check_valid_matrix(covariance_));

		Matrix cov = getCovariance();

		//covariance_matrix_min(cov,minimum_exploration_variance_);
		cov += Matrix::Identity(cov.rows(),cov.cols())*minimum_exploration_variance_;

		//covariance_ = regularizeTikhonov(covariance_);
		std::pair<Vector,Matrix> statistics = std::make_pair(mean_,cov.llt().matrixL());

		//Matrix debug = statistics.second * statistics.second.transpose() - cov;

		assert(check_valid_matrix(statistics.first));
		assert(check_valid_matrix(statistics.second));

		return statistics;
	}

	void scaleCovarianceMatrixToMaximum(void){
		if (maximum_standard_deviations_.sum() < std::numeric_limits<Scalar>::infinity()){

			//After doing this I figures out that this can be done in place.
			Matrix& correlationAndDeviationMatrix = covariance_;

			assert(check_valid_matrix(covariance_));

			//The standard deviations to the diagonal
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				correlationAndDeviationMatrix(i,i) = sqrt(correlationAndDeviationMatrix(i,i));
			}

			assert(check_valid_matrix(covariance_));

			//The correlations are stored to the off-diagonal entries (upper triangular; the lower triangular values are meaningles)
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				for(unsigned j = i + 1 ; j < (unsigned)correlationAndDeviationMatrix.cols(); j++){

					Scalar prodOfDeviations = correlationAndDeviationMatrix(i,i)*correlationAndDeviationMatrix(j,j);
					if (prodOfDeviations > std::numeric_limits<Scalar>::min()){
						correlationAndDeviationMatrix(i,j) /= (correlationAndDeviationMatrix(i,i)*correlationAndDeviationMatrix(j,j));
					}

				}
			}


			assert(check_valid_matrix(covariance_));

			//Cap the standard deviations to the maximums
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				correlationAndDeviationMatrix(i,i) = std::min(correlationAndDeviationMatrix(i,i),maximum_standard_deviations_(i));
			}

			assert(check_valid_matrix(covariance_));

			//Rebuild the upper triangle
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				for(unsigned j = i + 1 ; j < (unsigned)correlationAndDeviationMatrix.cols(); j++){
					covariance_(i,j) = correlationAndDeviationMatrix(i,j)*correlationAndDeviationMatrix(i,i)*correlationAndDeviationMatrix(j,j);
				}
			}

			assert(check_valid_matrix(covariance_));

			//Rebuild the diagonal
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				covariance_(i,i) = covariance_(i,i) * covariance_(i,i);
			}

			assert(check_valid_matrix(covariance_));

			//Copy the lower triangle
			for(unsigned i = 0; i < (unsigned)correlationAndDeviationMatrix.rows(); i++){
				for(unsigned j = 0 ; j < i; j++){
					covariance_(i,j) = correlationAndDeviationMatrix(j,i);
				}
			}

			assert(check_valid_matrix(covariance_));

		}
	}

	void computeDecomposition(void){
		sampling_decomposition_ = decompositionForSampling();
		decomposition_computed_ = true;
	}

	std::vector<Vector> sample(const unsigned& amount){

		std::vector<Vector> samples;
		samples.reserve(amount);

		if (!initialized_){
			return samples;
		}


		if (!decomposition_computed_){
			computeDecomposition();
		}

		//If the covariance matrix is singular or infinite.
		Scalar checksum = sampling_decomposition_.second.sum();
		if (checksum - checksum != checksum - checksum){
			return samples;
		}

		Vector tmp = Vector::Zero(mean_.size());
		Vector tmp_scaled = Vector::Zero(mean_.size());

		for (unsigned i = 0; i < amount; i++){
			tmp = BoxMuller<float>(mean_.size()); //Standard normal distribution
			assert(check_valid_matrix(tmp));

			tmp_scaled = sampling_decomposition_.second * tmp;
			assert(check_valid_matrix(tmp));

			tmp = sampling_decomposition_.first + tmp_scaled; //Distributed according to the current distribution
			assert(check_valid_matrix(tmp));

			samples.push_back(tmp);
		}

		return samples;
	}


	//std::vector<Vector> sampleConditionedToMean(const unsigned& amount,const unsigned int& conditioningDim){

	//	std::vector<Vector> samples;
	//	samples.reserve(amount);

	//	if (!initialized_){
	//		return samples;
	//	}

	//	//If the covariance matrix is singular or infinite, return just the mean.
	//	Scalar checksum = sampling_decomposition_.second.sum();
	//	if (checksum - checksum != checksum - checksum){
	//		return samples;
	//	}

	//	Vector tmp = Vector::Zero(mean_.size());

	//	for (unsigned i = 0; i < amount; i++){
	//		tmp = BoxMuller<float>(mean_.size()); //Standard normal distribution
	//		assert(check_valid_matrix(tmp));
	//		tmp = sampling_decomposition_.first + step_size_ * sampling_decomposition_.second * tmp; //Distributed according to the current distribution
	//		assert(check_valid_matrix(tmp));
	//		samples.push_back(tmp);
	//	}

	//	return samples;
	//}


	Vector getMean(){


		return mean_;


		//return Vector::Zero(sampling_decomposition_.first.size());

	}

	Matrix getCovariance(){

		assert(check_valid_matrix(covariance_));

		Matrix cov = covariance_;

		cov *= step_size_*step_size_;

		assert(check_valid_matrix(cov));

		//covariance_matrix_min(cov,minimum_exploration_variance_);

		assert(check_valid_matrix(cov));

		return cov;

	}


};


#endif