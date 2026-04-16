#ifndef BUILTIN_DATA_H
#define BUILTIN_DATA_H

#include "arrow/vendored/datetime/date.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

//adapted from apache gandiva
//source: https://github.com/apache/arrow/blob/3da66003ab2543c231fdf6551c2eb886f9a7e68f/cpp/src/gandiva/precompiled/epoch_time_point.h
//Apache-2.0 License
namespace {
   namespace date = arrow_vendored::date;
   class DateHelper {
   public:
      explicit DateHelper(std::chrono::nanoseconds nanosSinceEpoch)
         : tp(nanosSinceEpoch) {}

      explicit DateHelper(int64_t nanosecondsSinceEpoch)
         : DateHelper(std::chrono::nanoseconds(nanosecondsSinceEpoch)) {}

      int64_t tmYear() const { return static_cast<int>(yearMonthDay().year()) - 1900; }

      int64_t tmMon() const { return static_cast<unsigned int>(yearMonthDay().month()) - 1; }
      int64_t tmHour() const { return timeOfDay().hours().count(); }

      int64_t tmYday() const {
         auto toDays = date::floor<date::days>(tp);
         auto firstDayInYear = date::sys_days{
            yearMonthDay().year() / date::jan / 1};
         return (toDays - firstDayInYear).count();
      }

      int64_t tmMday() const { return static_cast<unsigned int>(yearMonthDay().day()); }

      DateHelper addMonths(int numMonths) const {
         auto ymd = yearMonthDay() + date::months(numMonths);
         return DateHelper((date::sys_days{ymd} + // NOLINT
                            timeOfDay().to_duration())
                              .time_since_epoch());
      }

      bool operator==(const DateHelper& other) const { return tp == other.tp; }

      int64_t nanosSinceEpoch() const { return tp.time_since_epoch().count(); }

   private:
      date::year_month_day yearMonthDay() const {
         return date::year_month_day{
            date::floor<date::days>(tp)}; // NOLINT
      }

      date::time_of_day<std::chrono::nanoseconds> timeOfDay() const {
         auto nanosSinceMidnight =
            tp - date::floor<date::days>(tp);
         return date::time_of_day<std::chrono::nanoseconds>(
            nanosSinceMidnight);
      }

      std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp;
   };
}
//end adapted from apache gandiva

namespace builtin::date {
   static constexpr int64_t NANOS_PER_SECOND = 1000000000LL;
   static constexpr int64_t SECONDS_PER_DAY = 86400LL;
   static constexpr int64_t NANOS_PER_DAY = NANOS_PER_SECOND * SECONDS_PER_DAY;

   int64_t subtractMonths(int64_t date, int64_t months) {
      return DateHelper(date).addMonths(-months).nanosSinceEpoch();
   }
   int64_t addMonths(int64_t nanos, int64_t months) {
      return DateHelper(nanos).addMonths(months).nanosSinceEpoch();
   }
   int64_t extractYear(int64_t date) {
      return DateHelper(date).tmYear() + 1900;
   }
   int64_t extractMonth(int64_t date) {
      return DateHelper(date).tmMon() + 1;
   }
   int64_t extractDay(int64_t date) {
      return DateHelper(date).tmMday();
   }
   int64_t dateDiffSeconds(int64_t start, int64_t end) {
      auto diffNanos=end-start;
      return diffNanos/(1000000000ull);
   }
   int64_t extractHour(int64_t date) {
      return DateHelper(date).tmHour();
   }
   std::int64_t parseDateToNanoseconds(const std::string& dateStr) {
      std::istringstream ss(dateStr);
      std::chrono::sys_time<std::chrono::nanoseconds> tp;
      if (dateStr.size() == 10) {
         ss >> arrow_vendored::date::parse("%Y-%m-%d", tp);
      } else {
         ss >> arrow_vendored::date::parse("%Y-%m-%dT%H:%M:%S", tp);
      }
      if (ss.fail()) {
         // Handle parsing error
         return -1;
      }
      return tp.time_since_epoch().count();
   }

   int64_t fromYMD(int64_t year, int64_t month, int64_t day) {
      auto ymd = arrow_vendored::date::year{static_cast<int>(year)} /
                 arrow_vendored::date::month{static_cast<unsigned>(month)} /
                 arrow_vendored::date::day{static_cast<unsigned>(day)};
      arrow_vendored::date::sys_days sd{ymd};
      auto ns = std::chrono::nanoseconds{sd.time_since_epoch()};
      return ns.count();
   }

   int64_t today() {
      using namespace std::chrono;
      auto now = system_clock::now();
      auto nsSinceEpoch = duration_cast<nanoseconds>(now.time_since_epoch()).count();
      // Truncate to midnight UTC.
      return (nsSinceEpoch / NANOS_PER_DAY) * NANOS_PER_DAY;
   }

   int64_t fromIsoFormat(const std::string& s) {
      return parseDateToNanoseconds(s);
   }

   int64_t diff(int64_t lhs, int64_t rhs) {
      // Return difference as an interval encoded in nanoseconds.
      return lhs - rhs;
   }

   int64_t addInterval(int64_t d, int64_t interval_nanos) {
      return d + interval_nanos;
   }

   int64_t subInterval(int64_t d, int64_t interval_nanos) {
      return d - interval_nanos;
   }

   std::string toString(int64_t nanos) {
      auto sd = arrow_vendored::date::sys_days{} +
                std::chrono::floor<arrow_vendored::date::days>(std::chrono::nanoseconds{nanos});
      auto ymd = arrow_vendored::date::year_month_day{sd};
      std::ostringstream ss;
      ss << std::setfill('0')
         << std::setw(4) << static_cast<int>(ymd.year()) << '-'
         << std::setw(2) << static_cast<unsigned>(ymd.month()) << '-'
         << std::setw(2) << static_cast<unsigned>(ymd.day());
      return ss.str();
   }

   // Python datetime.date.weekday(): Monday == 0, Sunday == 6
   int64_t weekday(int64_t nanos) {
      auto sd = arrow_vendored::date::sys_days{} +
                std::chrono::floor<arrow_vendored::date::days>(std::chrono::nanoseconds{nanos});
      auto wd = arrow_vendored::date::weekday{sd};
      // iso_encoding: Monday==1 ... Sunday==7
      return static_cast<int64_t>(wd.iso_encoding()) - 1;
   }

   int64_t isoWeekday(int64_t nanos) {
      return weekday(nanos) + 1;
   }

   // Python's proleptic Gregorian ordinal: date(1,1,1).toordinal() == 1
   int64_t toOrdinal(int64_t nanos) {
      auto sd = arrow_vendored::date::sys_days{} +
                std::chrono::floor<arrow_vendored::date::days>(std::chrono::nanoseconds{nanos});
      // Epoch 1970-01-01 toordinal is 719163
      constexpr int64_t EPOCH_ORDINAL = 719163;
      return sd.time_since_epoch().count() + EPOCH_ORDINAL;
   }

   bool eq(int64_t a, int64_t b) { return a == b; }
   bool neq(int64_t a, int64_t b) { return a != b; }
   bool lt(int64_t a, int64_t b) { return a < b; }
   bool lte(int64_t a, int64_t b) { return a <= b; }
   bool gt(int64_t a, int64_t b) { return a > b; }
   bool gte(int64_t a, int64_t b) { return a >= b; }
}

namespace builtin::interval {
   static constexpr int64_t NANOS_PER_SECOND = 1000000000LL;
   static constexpr int64_t SECONDS_PER_DAY = 86400LL;
   static constexpr int64_t NANOS_PER_DAY = NANOS_PER_SECOND * SECONDS_PER_DAY;

   int64_t fromDaysSeconds(int64_t days, int64_t seconds) {
      return days * NANOS_PER_DAY + seconds * NANOS_PER_SECOND;
   }

   // Python datetime.timedelta.days: floor division of total seconds by 86400.
   int64_t days(int64_t interval_nanos) {
      int64_t totalSeconds = interval_nanos / NANOS_PER_SECOND;
      // floor division
      int64_t d = totalSeconds / SECONDS_PER_DAY;
      int64_t r = totalSeconds % SECONDS_PER_DAY;
      if (r < 0) {
         d -= 1;
      }
      return d;
   }

   // Python datetime.timedelta.seconds: always in [0, 86400).
   int64_t seconds(int64_t interval_nanos) {
      int64_t totalSeconds = interval_nanos / NANOS_PER_SECOND;
      int64_t r = totalSeconds % SECONDS_PER_DAY;
      if (r < 0) {
         r += SECONDS_PER_DAY;
      }
      return r;
   }

   double totalSeconds(int64_t interval_nanos) {
      return static_cast<double>(interval_nanos) / static_cast<double>(NANOS_PER_SECOND);
   }

   int64_t add(int64_t a, int64_t b) { return a + b; }
   int64_t sub(int64_t a, int64_t b) { return a - b; }
   int64_t neg(int64_t a) { return -a; }

   bool eq(int64_t a, int64_t b) { return a == b; }
   bool neq(int64_t a, int64_t b) { return a != b; }
   bool lt(int64_t a, int64_t b) { return a < b; }
   bool lte(int64_t a, int64_t b) { return a <= b; }
   bool gt(int64_t a, int64_t b) { return a > b; }
   bool gte(int64_t a, int64_t b) { return a >= b; }
   bool isNonzero(int64_t a) { return a != 0; }

   std::string toString(int64_t interval_nanos) {
      // Mimic Python datetime.timedelta.__str__: "[D day[s], ][H]H:MM:SS"
      int64_t d = days(interval_nanos);
      int64_t s = seconds(interval_nanos);
      int64_t h = s / 3600;
      int64_t m = (s / 60) % 60;
      int64_t sec = s % 60;
      std::ostringstream ss;
      if (d != 0) {
         ss << d << (d == 1 || d == -1 ? " day, " : " days, ");
      }
      ss << h << ':' << std::setfill('0') << std::setw(2) << m << ':' << std::setw(2) << sec;
      return ss.str();
   }
}

#endif //BUILTIN_DATA_H
