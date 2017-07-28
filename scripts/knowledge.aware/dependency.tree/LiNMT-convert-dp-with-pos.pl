#!/usr/bin/perl

my $line_num = 0;
my $first = 0;
while (my $line = <STDIN>) {
  ++$line_num;

  $line =~ s/[\r\n]//g;
  if ($line =~ /^$/) {
    print "\n";
    $first = 0;
    next;
  }

  my @words = split /\s+/,$line;
  if (scalar(@words) ne 5 and scalar(@words) ne 4) {
    print STDERR "\nformat error\n";
    print STDERR "$line $line_num\n";
  } else {
    if ($first ne 0) {
      print " ";
    }

    if (scalar(@words) eq 4) {

      if ($words[1] =~ /^(.*)\/(.*)$/) {
        $words[4] = $words[3];
        $words[3] = $words[2];
        $words[2] = $2;
        if( $1 eq "") {
          $words[1] = "/";
        } else {
          $words[1] = $1;
        }
      } else {
        print STDERR "\nformat error\n";
        print STDERR "$line $line_num\n";
      }
    }


    my $current = $words[0] - 1;
    print $words[4]." ".$words[2]." ".$current." ".$words[3];
    ++$first;
  }  

  if ($line_num % 10000 == 0) {
    print STDERR "\r$line_num sentences";
  }
}
print STDERR "\r$line_num sentences\n";




