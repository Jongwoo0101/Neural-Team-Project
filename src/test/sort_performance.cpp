/*
const string input_filename
1. 설명
sort_performance.cpp
loss_acc_log.txt 파일을 읽어 FINAL ACC와 loss를 기준으로 각각 정렬한 후, 원본 로그 형식 그대로 sort_acc.txt와 sort_loss.txt 두 파일에 기록
:cout에 대해서, 한글로 출력시 깨지는 문제가 있어 모두 영어로 구성
2-1. 실행
F6을 눌러도 정상 실행되도록 파일 위치를 세팅했다.
이전의 방식:
실행시 /output폴더로 들어간다. 따라서 src\test>위치에서 "sort_performance.exe"를 실행해야 장상적으로 작동한다.
cd ..
output\"sort_performance.exe"
2-2. 코드 수정: 코드 수정시 2번 방법으로 실행시 src\test>위치에서 아래 코드로 .exe 파일을 갱신해야한다.
g++ sort_performance.cpp -o sort_performance.exe
또는 가장 확실한 방법 중 하나로 실행시에 생성되는 code.exe파일을 제거하고 다시 실행하는 것
4. bits/stdc++.h 헤더파일을 사용하지 않는 경우의 사용해야하는 include헤더파일
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <functional>
5. 변경 사항
find_min_performance.cpp에서 sort_performance.cpp으로 이름이 변경되었다.
모델이 잘 훈련되었는지에 대한 지표로써 loss와 accuracy의 사용이 확정됨에 따라 다음과 같이 변경한다.
output_log.txt에서 loss_acc_log.txt으로 이름 변경
output.bat에서 loss.bat으로 이름 변경
6. 수정 요구사항

*/
#include <bits/stdc++.h>
using namespace std;

// 로그 한 줄의 데이터를 저장할 구조체
struct LogEntry
{
    string original_line;
    double loss;
    double acc;
    // 백분율 분석을 위한 하이퍼파라미터 추출
    string optimizer;
    string lr;
    int batch;
    int iters;
    int depth;
    string act_init;
    double l2;
};
/**
 * @brief 주어진 로그 라인에서 FINAL LOSS와 Test ACC 값을 파싱.
 * @param line 로그 한 줄 문자열
 * @return LogEntry 구조체
 */
LogEntry parse_log_line(const string &line)
{
    LogEntry entry;
    entry.original_line = line;

    // --- 1. 하이퍼파라미터 추출 ---
    // 파이썬 포맷: Key: Value,  또는 Key: Value

    // Helper function to extract and trim string
    auto extract_and_trim = [&](const string &keyword, size_t &start, size_t end_char_pos) -> string
    {
        // 1. 키워드 검색: 로그 라인에서 "Optimizer: " 또는 "Batch: " 같은 키워드가 시작하는 위치
        size_t pos = line.find(keyword);
        if (pos == string::npos)
            throw runtime_error("Missing keyword: " + keyword);
        
        // 2. 값 시작 위치 설정: 키워드 길이만큼 건너뛰어 값의 시작 위치를 설정
        start = pos + keyword.length();

        // 3. 값 끝 위치 설정: 다음 콤마(',') 또는 파이프('|')를 찾아 값의 끝을 결정
        size_t end = line.find(end_char_pos == string::npos ? ',' : '|', start);
        if (end == string::npos)
            end = line.length();
        
        // 4. 값 추출 및 공백 제거: 키워드와 끝 위치 사이의 문자열(값)을 추출하고 앞뒤 공백을 제거합니다.
        string value = line.substr(start, end - start);
        value.erase(0, value.find_first_not_of(' ')); // Trim leading spaces
        value.erase(value.find_last_not_of(' ') + 1); // Trim trailing spaces
        return value;
    };

    // py파일의 output을 저장하는 txt에서의 왼쪽 값이 첫 번째 매개변수
    // ex) Optimizer: Adam에서 'Optimizer:' 부분
    size_t start_pos;

    // Optimizer
    entry.optimizer = extract_and_trim("Optimizer: ", start_pos, string::npos);

    // LR
    entry.lr = extract_and_trim("LR: ", start_pos, string::npos);

    // Batch
    try
    {
        entry.batch = stoi(extract_and_trim("Batch: ", start_pos, string::npos));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse Batch value.");
    }

    // Iters
    try
    {
        entry.iters = stoi(extract_and_trim("Iters: ", start_pos, string::npos));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse Iters value.");
    }

    // Depth
    try
    {
        entry.depth = stoi(extract_and_trim("Depth: ", start_pos, string::npos));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse Depth value.");
    }

    // Act/Init
    entry.act_init = extract_and_trim("Act/Init: ", start_pos, string::npos);

    // L2 (참고: L2 뒤에 '|'가 온다.)
    try
    {
        entry.l2 = stod(extract_and_trim("L2: ", start_pos, '|'));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse L2 value.");
    }

    // --- 2. 결과 추출 ---

    // FINAL LOSS
    size_t loss_pos = line.find("FINAL LOSS:");
    if (loss_pos == string::npos)
        throw runtime_error("Missing FINAL LOSS.");
    loss_pos += 12; // Skip "FINAL LOSS: "
    size_t loss_end = line.find('|', loss_pos);
    try
    {
        entry.loss = stod(line.substr(loss_pos, loss_end - loss_pos));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse FINAL LOSS value.");
    }

    // Test ACC
    size_t acc_pos = line.find("Test ACC:");
    if (acc_pos == string::npos)
        throw runtime_error("Missing Test ACC.");
    acc_pos += 10;                             // "Test ACC: " 넘김
    size_t acc_end = line.find('\n', acc_pos); // 줄이 끝일 수 있으니 \n찾음
    if (acc_end == string::npos)
        acc_end = line.length();
    try
    {
        entry.acc = stod(line.substr(acc_pos, acc_end - acc_pos));
    }
    catch (...)
    {
        throw runtime_error("Failed to parse Test ACC value.");
    }

    return entry;
}


/**
 * @brief 정확도 내림차순 비교 함수
 */
bool compareByAcc(const LogEntry &a, const LogEntry &b)
{
    return a.acc > b.acc;
}

/**
 * @brief 손실 내림차순 비교 함수 (손실은 오름차순이 좋으므로, 작은 값이 앞에 오도록)
 */
bool compareByLoss(const LogEntry &a, const LogEntry &b)
{
    return a.loss < b.loss;
}

/**
 * @brief 하이퍼파라미터 분석 및 보고서: 상위 N% 데이터의 하이퍼파라미터 분포를 분석하고 포맷팅.
 * @param sorted_data 분석할 데이터 (정확도 내림차순 또는 손실 오름차순)
 * @param total_count 전체 데이터 개수
 * @param percentiles 분석할 백분율 목록
 * @param analysis_standard 분석 기준 (예: "Test ACC" 또는 "FINAL LOSS")
 * @param best_value 해당 기준의 최고 성능 값 (정확도 최대값, 손실 최소값)
 * @param ofs 결과를 기록할 출력 파일 스트림
 */
void analyze_top_percentiles(const vector<LogEntry> &sorted_data,
                             size_t total_count,
                             const vector<int> &percentiles,
                             const string &analysis_standard,
                             double best_value,
                             ofstream &ofs)
{
    // 1e-n 형식으로 변환하고 불필요한 부호나 소수점을 제거한다.
    auto format_scientific = [](double value) -> string
    {
        // Handle L2=0 case
        if (abs(value) < 1e-10)
            return "0";

        stringstream ss;
        // 과학적 표기법(scientific) 사용, 정밀도 0을 주어 1.e-08 형태를 만든다.
        ss << scientific << setprecision(0) << value;
        string s = ss.str();

        // '1.e' 형태의 '.' 제거 (setprecision(0) 때문에 발생할 수 있음)
        size_t dot_pos = s.find('.');
        if (dot_pos != string::npos && dot_pos + 1 < s.length() && s[dot_pos + 1] == 'e')
        {
            s.erase(dot_pos, 1);
        }

        // 지수부의 불필요한 선행 '0' 제거 (e.g., '1e-08' -> '1e-8')
        size_t e_pos = s.find('e');
        if (e_pos != string::npos && s.length() > e_pos + 2)
        {
            // e+ 또는 e- 뒤에 0이 있다면
            char sign = s[e_pos + 1];
            if ((sign == '+' || sign == '-') && s[e_pos + 2] == '0' && s.length() > e_pos + 3)
            {
                s.erase(e_pos + 2, 1); // 세 번째 문자(0) 제거
            }
        }
        return s;
    };
    // --- 헤더 출력 ---
    ofs << "=========================================================================" << "\n";
    ofs << "== Hyperparameter Occurrence Frequency Analysis (Standard: " << analysis_standard << ") ==" << "\n";
    ofs << "=========================================================================" << "\n";
    // 최고 성능 값을 출력
    ofs << "Best model" << analysis_standard << ": " << fixed << setprecision(4) << best_value << "\n\n";
    // 분석 로직은 이전 코드와 동일하게 유지
    // std::pair의 첫 번째 요소는 분석 파일에 표시될 하이퍼파라미터의 이름
    // std::pair의 두 번째 요소는 하이퍼파라미터의 실제 값을 추출, 값을 추출하는 람다 함수
    vector<pair<string, function<string(const LogEntry &)>>> hparam_getters = {

        {"Iters", [](const LogEntry &e)
         { return to_string(e.iters); }},
        {"Opt", [](const LogEntry &e)
         { return e.optimizer; }},
        {"LR", [&](const LogEntry &e)
         {
             try
             {
                 return format_scientific(stod(e.lr));
             }
             catch (...)
             {
                 return e.lr; // 구문 분석 오류, 원래 문자열을 반환
             }
         }},
        {"Depth", [](const LogEntry &e)
         { return to_string(e.depth); }},
        {"Act/Init", [](const LogEntry &e)
         { return e.act_init; }},
        {"L2", [&](const LogEntry &e)
         {
             return format_scientific(e.l2);
         }},
        {"Batch", [](const LogEntry &e)
         { return to_string(e.batch); }},
    };

    map<string, map<string, vector<int>>> analysis_results;
    for (const auto &[name, getter] : hparam_getters)
    {
        analysis_results[name] = {};
    }

    // 각 백분율별로 데이터 수집 및 집계
    for (size_t p_idx = 0; p_idx < percentiles.size(); ++p_idx)
    {
        int p = percentiles[p_idx];
        int top_k = static_cast<int>(ceil(total_count * p / 100.0));
        top_k = min(top_k, (int)sorted_data.size());

        for (int i = 0; i < top_k; ++i)
        {
            const LogEntry &entry = sorted_data[i];
            for (const auto &[name, getter] : hparam_getters)
            {
                string value = getter(entry);
                if (analysis_results[name].find(value) == analysis_results[name].end())
                {
                    analysis_results[name][value].resize(percentiles.size(), 0);
                }
                analysis_results[name][value][p_idx]++;
            }
        }
    }

    // --- 결과 포맷 출력 ---
    ofs << "Top Percentiles: {";
    for (size_t i = 0; i < percentiles.size(); ++i)
    {
        ofs << percentiles[i] << "%" << (i < percentiles.size() - 1 ? ", " : "");
    }
    ofs << "}\n";
    ofs << "Total Entries: " << total_count << "\n";
    ofs << "---\n";

    // 분석 결과 출력
    for (const auto &[hparam_name, hparam_values] : analysis_results)
    {
        ofs << hparam_name << ": ";

        // 후보 값 개수를 세기 위한 카운터
        int item_cnt = 0;
        // 후보 값 목록을 순회
        for (const auto &[value, cnt] : hparam_values)
        {
            //후보값 수가 2개 이상이면 2개씩 출력하고 줄바꿈 (첫 번째 항목은 줄바꿈 하지 않음)
            if (item_cnt > 0 && item_cnt % 2 == 0)
            {
                // 줄바꿈 후 인덴테이션 (HPARAM_NAME: 뒤에 맞추기 위해 8칸 공백 사용)
                ofs << "\n"<< "        ";
            }
            ofs << "[" << value << ": {";
            for (size_t p_idx = 0; p_idx < percentiles.size(); ++p_idx)
            {
                int p = percentiles[p_idx];
                int top_k = static_cast<int>(ceil(total_count * p / 100.0));
                top_k = max(1, top_k);

                double percentage = (double)cnt[p_idx] / top_k * 100.0;

                ofs << fixed << setprecision(1) << percentage << "%" << (p_idx < percentiles.size() - 1 ? ", " : "");
            }
            ofs << "}] ";
            item_cnt++;
        }
        ofs << "\n";
    }
    ofs << "\n"; // 두 섹션 사이 공백 추가
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // hyperparameter_tuning_loss.py의 결과
    const string input_filename = "../../target/loss_acc_adam.txt";
    // test/sort_performance.cpp에서 ../log/ 디렉토리 안에 파일을 생성
    const string output_dir = "../../log/";
    const string acc_output_filename = output_dir + "sort_acc.txt";
    const string loss_output_filename = output_dir + "sort_loss.txt";
    const string percentiles_output_filename = output_dir + "sort_percentiles.txt";

    // 1. 파일 읽기
    ifstream ifs(input_filename);
    if (!ifs.is_open())
    {
        cerr << "Error: Unable to open input file. Check path: " << input_filename << '\n';
        return 1;
    }

    vector<LogEntry> data;
    string line;
    while (getline(ifs, line))
    {
        if (!line.empty())
        {
            // (안전한 parse_log_line 함수를 호출하여 데이터를 LogEntry 벡터에 추가)
            try
            {
                data.push_back(parse_log_line(line));
            }
            catch (const exception &e)
            {
                cerr << "Error parsing line: " << line << " | Reason: " << e.what() << '\n';
                continue;
            }
        }
    }
    ifs.close();

    if (data.empty())
    {
        cout << input_filename << ":No data found or all data failed to parse.\n";
        return 0;
    }

    // total_count는 main 함수 내에서 정의하고 analyze_top_percentiles로 전달해야 함
    size_t total_count = data.size();
    // 상위 1,3,6,12%를 잡기 위한 값
    vector<int> percentiles = {1, 3, 6, 12};
    // 2. 데이터 정렬
    // ACC 내림차순 정렬 (높은 ACC가 상위)
    vector<LogEntry> acc_sorted_data = data;
    sort(acc_sorted_data.begin(), acc_sorted_data.end(), compareByAcc);

    // LOSS 오름차순 정렬 (낮은 LOSS가 상위)
    vector<LogEntry> loss_sorted_data = data;
    sort(loss_sorted_data.begin(), loss_sorted_data.end(), compareByLoss);

    // --- 최고 성능 값 추출 --- 0에 최고 성능 값이 있다.
    double best_acc = acc_sorted_data[0].acc;
    double best_loss = loss_sorted_data[0].loss;

    // 3. 정렬된 파일 기록 (sort_acc.txt, sort_loss.txt)

    // sort_acc.txt 기록
    ofstream acc_ofs(acc_output_filename);
    if (acc_ofs.is_open())
    {
        for (const auto &entry : acc_sorted_data)
        {
            acc_ofs << entry.original_line << "\n";
        }
        acc_ofs.close();
        cout << "Successfully written to " << acc_output_filename << '\n';
    }
    else
    {
        cerr << "Error: Unable to open " << acc_output_filename << ". Check if the directory exists." << '\n';
    }

    // sort_loss.txt 기록
    ofstream loss_ofs(loss_output_filename);
    if (loss_ofs.is_open())
    {
        for (const auto &entry : loss_sorted_data)
        {
            loss_ofs << entry.original_line << "\n";
        }
        loss_ofs.close();
        cout << "Successfully written to " << loss_output_filename << '\n';
    }
    else
    {
        cerr << "Error: Unable to open " << loss_output_filename << ". Check if the directory exists.\n";
    }

    // 4. 백분율 분석 및 기록 (sort_percentiles.txt)
    // ofstream을 열 때 std::ios::out 플래그가 기본값이므로 파일이 새로 생성되거나 덮어쓰기 된다.
    ofstream percentiles_ofs(percentiles_output_filename);
    if (!percentiles_ofs.is_open())
    {
        cerr << "Error: Unable to open " << percentiles_output_filename << ". Check if the directory exists: " << output_dir << '\n';
        return 1;
    }
    // 4-1. ACC 기준 분석 수행 (ACC 내림차순 데이터 사용)
    // analyze_top_percentiles(acc_sorted_data, total_count, percentiles, "Test ACC", percentiles_ofs);
    analyze_top_percentiles(acc_sorted_data, total_count, percentiles, "Test ACC", best_acc, percentiles_ofs);

    // 4-2. LOSS 기준 분석 수행 (LOSS 오름차순 데이터 사용)
    // analyze_top_percentiles(loss_sorted_data, total_count, percentiles, "FINAL LOSS", percentiles_ofs);
    analyze_top_percentiles(loss_sorted_data, total_count, percentiles, "FINAL LOSS", best_loss, percentiles_ofs);

    percentiles_ofs.close();
    cout << "Successfully written to " << percentiles_output_filename << '\n';
}